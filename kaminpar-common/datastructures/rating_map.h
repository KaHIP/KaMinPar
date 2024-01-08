/*******************************************************************************
 * @file:   rating_map.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Map to compute gain values. The specific implementation depends on
 * the number of target blocks.
 ******************************************************************************/
#pragma once

#include "kaminpar-common/datastructures/fast_reset_array.h"
#include "kaminpar-common/datastructures/fixed_size_sparse_map.h"
#include "kaminpar-common/datastructures/sparse_map.h"

namespace kaminpar {
template <typename Value, typename Key, std::size_t kSuperSmallVectorSize = 128>
class BlockRatingMap {
  using SmallMap = FixedSizeSparseMap<Key, Value>;

public:
  enum class MapType {
    SUPER_SMALL,
    SMALL,
    LARGE
  };

  explicit BlockRatingMap(const std::size_t max_size)
      : _max_size(max_size),
        _super_small_map(kSuperSmallVectorSize) {}

  MapType update_upper_bound(const std::size_t upper_bound_size) {
    select_map(upper_bound_size);
    return _selected_map;
  }

  template <typename Lambda>
  decltype(auto) execute(const std::size_t upper_bound, Lambda &&lambda) {
    update_upper_bound(upper_bound);

    switch (_selected_map) {
    case MapType::SUPER_SMALL:
      return lambda(_super_small_map);
    case MapType::SMALL:
      return lambda(_small_map);
    case MapType::LARGE:
      return lambda(_large_map);
    }

    __builtin_unreachable();
  }

  [[nodiscard]] std::size_t small_map_counter() const {
    return _small_map_counter;
  }

  [[nodiscard]] std::size_t super_small_map_counter() const {
    return _super_small_map_counter;
  }

  [[nodiscard]] std::size_t large_map_counter() const {
    return _large_map_counter;
  }

  [[nodiscard]] std::size_t max_size() const {
    return _max_size;
  }

  void change_max_size(const std::size_t max_size) {
    _max_size = max_size;
  }

private:
  void select_map(const std::size_t upper_bound_size) {
    if (upper_bound_size < kSuperSmallVectorSize) {
      _selected_map = MapType::SUPER_SMALL;
      ++_super_small_map_counter;
    } else if (_max_size < SmallMap::MAP_SIZE || upper_bound_size > SmallMap::MAP_SIZE / 3) {
      _selected_map = MapType::LARGE;
      ++_large_map_counter;
    } else {
      _selected_map = MapType::SMALL;
      ++_small_map_counter;
    }

    if (_selected_map == MapType::LARGE && _large_map.capacity() < _max_size) {
      _large_map.resize(_max_size);
    }
  }

  std::size_t _max_size;

  MapType _selected_map = MapType::SMALL;
  FastResetArray<Value, Key> _super_small_map;
  SmallMap _small_map{};
  FastResetArray<Value, Key> _large_map{}; // allocate on demand

  std::size_t _super_small_map_counter = 0;
  std::size_t _small_map_counter = 0;
  std::size_t _large_map_counter = 0;
};

template <typename Value, typename Key, typename LargeMap = FastResetArray<Value, Key>>
class RatingMap {
  using SuperSmallMap = FixedSizeSparseMap<Key, Value, 128>;
  using SmallMap = FixedSizeSparseMap<Key, Value>;

public:
  enum class MapType {
    SUPER_SMALL,
    SMALL,
    LARGE
  };

  explicit RatingMap(const std::size_t max_size) : _max_size{max_size} {}

  MapType update_upper_bound(const std::size_t upper_bound_size) {
    select_map(upper_bound_size);
    return _selected_map;
  }

  template <typename Lambda>
  decltype(auto) execute(const std::size_t upper_bound, Lambda &&lambda) {
    update_upper_bound(upper_bound);

    switch (_selected_map) {
    case MapType::SUPER_SMALL:
      return lambda(_super_small_map);
    case MapType::SMALL:
      return lambda(_small_map);
    case MapType::LARGE:
      return lambda(_large_map);
    }

    __builtin_unreachable();
  }

  [[nodiscard]] std::size_t small_map_counter() const {
    return _small_map_counter;
  }

  [[nodiscard]] std::size_t super_small_map_counter() const {
    return _super_small_map_counter;
  }

  [[nodiscard]] std::size_t large_map_counter() const {
    return _large_map_counter;
  }

  [[nodiscard]] std::size_t max_size() const {
    return _max_size;
  }

  void change_max_size(const std::size_t max_size) {
    _max_size = max_size;
  }

private:
  void select_map(const std::size_t upper_bound_size) {
    if (upper_bound_size < SuperSmallMap::MAP_SIZE / 3) {
      _selected_map = MapType::SUPER_SMALL;
      ++_super_small_map_counter;
    } else if (_max_size < SmallMap::MAP_SIZE || upper_bound_size > SmallMap::MAP_SIZE / 3) {
      _selected_map = MapType::LARGE;
      ++_large_map_counter;
    } else {
      _selected_map = MapType::SMALL;
      ++_small_map_counter;
    }

    if (_selected_map == MapType::LARGE && _large_map.capacity() < _max_size) {
      _large_map.resize(_max_size);
    }
  }

  std::size_t _max_size;

  MapType _selected_map = MapType::SMALL;
  SuperSmallMap _super_small_map{};
  SmallMap _small_map{};
  LargeMap _large_map{}; // allocate on demand

  std::size_t _small_map_counter = 0;
  std::size_t _super_small_map_counter = 0;
  std::size_t _large_map_counter = 0;
};
} // namespace kaminpar
