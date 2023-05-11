/*******************************************************************************
 * @file:   rating_map.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Map to compute gain values. The specific implementation depends on
 * the number of target blocks.
 ******************************************************************************/
#pragma once

#include "common/datastructures/fast_reset_array.h"
#include "common/datastructures/fixed_size_sparse_map.h"
#include "common/datastructures/sparse_map.h"

namespace kaminpar {
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

  MapType update_upper_bound_size(const std::size_t upper_bound_size) {
    select_map(upper_bound_size);
    return _selected_map;
  }

  template <typename F1, typename F2> decltype(auto) run_with_map(F1 &&f1, F2 &&f2) {
    switch (_selected_map) {
    case MapType::SUPER_SMALL:
      return f1(_super_small_map);
    case MapType::SMALL:
      return f1(_small_map);
    case MapType::LARGE:
      return f2(_large_map);
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
  MapType _selected_map{MapType::SMALL};
  SuperSmallMap _super_small_map{};
  SmallMap _small_map{};
  LargeMap _large_map{}; // allocate on demand

  std::size_t _small_map_counter{0};
  std::size_t _super_small_map_counter{0};
  std::size_t _large_map_counter{0};
};
} // namespace kaminpar
