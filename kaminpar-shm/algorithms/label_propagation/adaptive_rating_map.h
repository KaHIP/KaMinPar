/*******************************************************************************
 * Cache-adaptive rating accumulator.
 *
 * @file:   adaptive_rating_map.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>

#include "kaminpar-shm/algorithms/label_propagation/fixed_capacity_rating_map.h"
#include "kaminpar-shm/algorithms/label_propagation/linear_rating_map.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/fast_reset_array.h"
#include "kaminpar-common/datastructures/sparse_map.h"

namespace kaminpar::shm::lp {

namespace adaptive_rating_map {

template <typename Key, typename Value>
using FastResetArray = ::kaminpar::FastResetArray<Value, Key>;

template <typename Key, typename Value> using SparseMap = ::kaminpar::SparseMap<Key, Value>;

} // namespace adaptive_rating_map

template <
    typename Key,
    typename Value,
    template <typename, typename> typename DirectMap = adaptive_rating_map::FastResetArray>
class AdaptiveRatingMap {
public:
  static constexpr std::size_t kMaxHashCapacity = 32768;
  static constexpr std::size_t kMaxHashSize = kMaxHashCapacity / 3;
  static constexpr std::size_t kHashCapacityFactor = 8;
  static constexpr std::size_t kMinHashCapacity = 128;
  static constexpr std::size_t kDirectMapThreshold = 512;
  static constexpr std::size_t kLinearMapCapacity = 8;

  using LinearMap = LinearRatingMap<Key, Value, kLinearMapCapacity>;
  using HashMap = FixedCapacityRatingMap<Key, Value, kMaxHashCapacity>;

  explicit AdaptiveRatingMap(const std::size_t max_size)
      : _max_size(max_size),
        _hash_map(Value{}, max_size > kDirectMapThreshold) {
    KASSERT(
        max_size > std::size_t{0},
        "adaptive rating maps require a non-empty key universe",
        assert::always
    );
    if (use_direct_map()) {
      ensure_direct_map_capacity();
    }
  }

  AdaptiveRatingMap(const AdaptiveRatingMap &) = delete;
  AdaptiveRatingMap &operator=(const AdaptiveRatingMap &) = delete;
  AdaptiveRatingMap(AdaptiveRatingMap &&) noexcept = default;
  AdaptiveRatingMap &operator=(AdaptiveRatingMap &&) noexcept = default;

  template <typename Lambda>
  decltype(auto) execute(const std::size_t upper_bound, Lambda &&lambda) {
    if (use_direct_map()) {
      return lambda(_direct_map);
    }

    if (upper_bound <= kLinearMapCapacity) {
      return lambda(_linear_map);
    }

    if (upper_bound <= kMaxHashSize) {
      return lambda(prepare_hash_map_unchecked(upper_bound));
    }

    ensure_direct_map_capacity();
    return lambda(_direct_map);
  }

  [[nodiscard]] HashMap &prepare_hash_map(const std::size_t upper_bound) {
    KASSERT(
        upper_bound <= kMaxHashSize,
        "too many entries for the fixed-capacity rating map",
        assert::always
    );
    return prepare_hash_map_unchecked(upper_bound);
  }

  [[nodiscard]] HashMap &hash_map() {
    return _hash_map;
  }

  [[nodiscard]] std::size_t max_size() const {
    return _max_size;
  }

  void change_max_size(const std::size_t max_size) {
    _max_size = max_size;
    if (use_direct_map()) {
      ensure_direct_map_capacity();
    }
  }

private:
  [[nodiscard]] HashMap &prepare_hash_map_unchecked(const std::size_t upper_bound) {
    const std::size_t capacity = hash_capacity(upper_bound);
    if (_hash_map.size() == 0) {
      if (_hash_map.capacity() != capacity) {
        _hash_map.set_capacity(capacity);
      }
    } else {
      KASSERT(_hash_map.capacity() >= capacity);
    }
    return _hash_map;
  }
  [[nodiscard]] bool use_direct_map() const {
    return _max_size <= kDirectMapThreshold;
  }

  [[nodiscard]] static std::size_t hash_capacity(const std::size_t upper_bound) {
    const std::size_t required_capacity =
        kHashCapacityFactor * std::max<std::size_t>(upper_bound, 1);
    return std::clamp<std::size_t>(
        std::bit_ceil(required_capacity), kMinHashCapacity, kMaxHashCapacity
    );
  }

  void ensure_direct_map_capacity() {
    if (_direct_map.capacity() < _max_size) {
      _direct_map.resize(_max_size);
    }
  }

  std::size_t _max_size;
  LinearMap _linear_map;
  HashMap _hash_map;
  DirectMap<Key, Value> _direct_map;
};

} // namespace kaminpar::shm::lp
