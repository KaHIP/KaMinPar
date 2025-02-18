/*******************************************************************************
 * Data structure to accummulate gain values: picks an appropriate data
 * structure depending on the (expected) number of adjacent blocks. For high
 * degree nodes, a backyard data structure is used (default: vector of size
 * O(n)).
 *
 * @file:   rating_map.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#ifdef KAMINPAR_SPARSEHASH_FOUND
#include <google/dense_hash_map>
#endif

#include <unordered_map>

#include "kaminpar-common/datastructures/fast_reset_array.h"
#include "kaminpar-common/datastructures/fixed_size_sparse_map.h"
#include "kaminpar-common/datastructures/sparse_map.h"

namespace kaminpar {

namespace rm_backyard {

template <typename Key, typename Value>
using FastResetArray = ::kaminpar::FastResetArray<Value, Key>;

template <typename Key, typename Value> using SparseMap = ::kaminpar::SparseMap<Key, Value>;

template <typename Key, typename Value> class UnorderedMap {
public:
  Value &operator[](const Key key) {
    return map[key];
  }

  [[nodiscard]] auto &entries() {
    return map;
  }

  void clear() {
    map.clear();
  }

  [[nodiscard]] std::size_t capacity() const {
    return std::numeric_limits<std::size_t>::max();
  }

  void resize(std::size_t) {}

private:
  std::unordered_map<Key, Value> map;
};

#ifdef KAMINPAR_SPARSEHASH_FOUND
template <typename Key, typename Value> class Sparsehash {
public:
  Sparsehash() {
    map.set_empty_key(std::numeric_limits<Key>::max());
  }

  Value &operator[](const Key key) {
    return map[key];
  }

  [[nodiscard]] auto &entries() {
    return map;
  }

  void clear() {
    map.clear();
  }

  [[nodiscard]] std::size_t capacity() const {
    return std::numeric_limits<std::size_t>::max();
  }

  void resize(std::size_t) {}

private:
  google::dense_hash_map<Key, Value> map;
};
#else  // KAMINPAR_SPARSEHASH_FOUND
template <typename Key, typename Value> using Sparsehash = SparseMap<Key, Value>;
#endif // KAMINPAR_SPARSEHASH_FOUND

} // namespace rm_backyard

template <
    typename Value,
    typename Key,
    template <typename, typename> typename LargeMap = rm_backyard::FastResetArray>
class RatingMap {
  SET_STATISTICS_FROM_GLOBAL();

  using SuperSmallMap = FixedSizeSparseMap<Key, Value, 128>;
  using SmallMap = FixedSizeSparseMap<Key, Value>;

public:
  enum class MapType {
    SUPER_SMALL,
    SMALL,
    LARGE
  };

  explicit RatingMap(const std::size_t max_size = 0) : _max_size(max_size) {}

  RatingMap(const RatingMap &) = delete;
  RatingMap &operator=(const RatingMap &) = delete;

  RatingMap(RatingMap &&) noexcept = default;
  RatingMap &operator=(RatingMap &&) noexcept = default;

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

  [[nodiscard]] SmallMap &small_map() {
    return _small_map;
  }

  [[nodiscard]] std::size_t small_map_counter() const {
    return kStatistics ? _small_map_counter : std::numeric_limits<std::size_t>::max();
  }

  [[nodiscard]] std::size_t super_small_map_counter() const {
    return kStatistics ? _super_small_map_counter : std::numeric_limits<std::size_t>::max();
  }

  [[nodiscard]] std::size_t large_map_counter() const {
    return kStatistics ? _large_map_counter : std::numeric_limits<std::size_t>::max();
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
      IFSTATS(++_super_small_map_counter);
    } else if (_max_size < SmallMap::MAP_SIZE || upper_bound_size > SmallMap::MAP_SIZE / 3) {
      _selected_map = MapType::LARGE;
      IFSTATS(++_large_map_counter);
    } else {
      _selected_map = MapType::SMALL;
      IFSTATS(++_small_map_counter);
    }

    if (_selected_map == MapType::LARGE && _large_map.capacity() < _max_size) {
      _large_map.resize(_max_size);
    }
  }

  std::size_t _max_size;

  MapType _selected_map = MapType::SMALL;

  // Small maps always get allocated
  SuperSmallMap _super_small_map{};
  SmallMap _small_map{};

  // Large map is only allocated on demand
  LargeMap<Key, Value> _large_map{};

  std::size_t _small_map_counter = 0;
  std::size_t _super_small_map_counter = 0;
  std::size_t _large_map_counter = 0;
};

} // namespace kaminpar
