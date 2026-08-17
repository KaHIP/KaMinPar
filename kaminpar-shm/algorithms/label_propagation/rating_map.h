/*******************************************************************************
 * Local maps for neighborhood rating aggregation.
 *
 * @file:   rating_map.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/fast_reset_array.h"
#include "kaminpar-common/datastructures/sparse_map.h"
#include "kaminpar-common/math.h"

namespace kaminpar::shm::lp {

template <typename Key, typename Value, std::size_t Capacity> class LinearRatingMap {
  static_assert(Capacity > 0);

  struct Element {
    Key key;
    Value value;
  };

public:
  explicit LinearRatingMap(const Value initial_value = Value()) : _initial_value(initial_value) {}

  [[nodiscard]] std::size_t capacity() const {
    return Capacity;
  }

  [[nodiscard]] std::size_t size() const {
    return _size;
  }

  [[nodiscard]] Element *begin() {
    return _entries.data();
  }

  [[nodiscard]] Element *end() {
    return _entries.data() + _size;
  }

  [[nodiscard]] const Element *begin() const {
    return _entries.data();
  }

  [[nodiscard]] const Element *end() const {
    return _entries.data() + _size;
  }

  [[nodiscard]] LinearRatingMap &entries() {
    return *this;
  }

  [[nodiscard]] const LinearRatingMap &entries() const {
    return *this;
  }

  [[nodiscard]] bool contains(const Key key) const {
    return find(key) != _size;
  }

  [[nodiscard]] const Value &get(const Key key) const {
    const std::size_t position = find(key);
    KASSERT(position != _size, "key not in linear rating map");
    return _entries[position].value;
  }

  Value &operator[](const Key key) {
    const std::size_t position = find(key);
    if (position != _size) {
      return _entries[position].value;
    }

    KASSERT(_size < Capacity, "linear rating map is full");
    _entries[_size] = {.key = key, .value = _initial_value};
    return _entries[_size++].value;
  }

  void clear() {
    _size = 0;
  }

private:
  [[nodiscard]] std::size_t find(const Key key) const {
    for (std::size_t i = 0; i < _size; ++i) {
      if (_entries[i].key == key) {
        return i;
      }
    }
    return _size;
  }

  Value _initial_value;
  std::array<Element, Capacity> _entries;
  std::size_t _size = 0;
};

template <typename Key, typename Value, std::size_t MaxCapacity = 32768>
class FixedCapacityRatingMap {
  struct Element {
    Key key;
    Value value;
  };

  struct SparseElement {
    // Every probe checks the timestamp, whereas only occupied probes need the
    // dense index; keep the timestamp in the low bits.
    std::uint64_t metadata;
  };

  struct ProbeResult {
    SparseElement *slot;
    std::uint32_t dense_index;
    bool found;
  };

  static_assert(math::is_power_of_2(MaxCapacity));
  static_assert(MaxCapacity >= 2);
  static_assert(MaxCapacity <= std::numeric_limits<std::uint32_t>::max());
  static_assert(sizeof(SparseElement) == sizeof(std::uint64_t));

  [[nodiscard]] static constexpr std::uint64_t
  pack_sparse_metadata(const std::uint32_t dense_index, const std::uint32_t timestamp) {
    return timestamp | (static_cast<std::uint64_t>(dense_index) << 32);
  }

  [[nodiscard]] static constexpr std::uint32_t sparse_dense_index(const std::uint64_t metadata) {
    return static_cast<std::uint32_t>(metadata >> 32);
  }

  [[nodiscard]] static constexpr std::uint32_t sparse_timestamp(const std::uint64_t metadata) {
    return static_cast<std::uint32_t>(metadata);
  }

  [[nodiscard]] std::size_t hash(const Key key) const {
    constexpr std::uint64_t kGoldenRatio = 0x9e3779b97f4a7c15ULL;
    return (static_cast<std::uint64_t>(key) * kGoldenRatio) >> _hash_shift;
  }

public:
  class Accumulator {
  public:
    void add(const Key key, const Value value) {
      const ProbeResult result = _map->find(key, _dense);
      if (result.found) {
        _dense[result.dense_index].value += value;
      } else {
        _map->add_element(key, result.slot, _dense)->value += value;
      }
    }

    [[nodiscard]] std::size_t size() const {
      return _map->_size;
    }

  private:
    friend class FixedCapacityRatingMap;

    explicit Accumulator(FixedCapacityRatingMap *map) : _map(map), _dense(map->_dense.get()) {}

    FixedCapacityRatingMap *_map;
    Element *_dense;
  };

  explicit FixedCapacityRatingMap(
      const Value initial_value = Value(), const bool allocate_storage = true
  )
      : _initial_value(initial_value) {
    if (allocate_storage) {
      allocate();
    }
  }

  FixedCapacityRatingMap(const FixedCapacityRatingMap &) = delete;
  FixedCapacityRatingMap &operator=(const FixedCapacityRatingMap &) = delete;
  FixedCapacityRatingMap(FixedCapacityRatingMap &&) noexcept = default;
  FixedCapacityRatingMap &operator=(FixedCapacityRatingMap &&) noexcept = default;

  [[nodiscard]] std::size_t capacity() const {
    return _capacity;
  }

  [[nodiscard]] bool is_allocated() const {
    return _sparse != nullptr;
  }

  [[nodiscard]] std::size_t size() const {
    return _size;
  }

  [[nodiscard]] Accumulator accumulator() {
    KASSERT(is_allocated());
    return Accumulator(this);
  }

  void set_capacity(const std::size_t capacity) {
    KASSERT(
        _size == std::size_t{0}, "cannot change the capacity of a non-empty map", assert::always
    );
    KASSERT(
        capacity <= MaxCapacity,
        "capacity exceeds the fixed-capacity rating map allocation",
        assert::always
    );
    allocate();
    _capacity = std::bit_ceil(std::max<std::size_t>(capacity, 2));
    _hash_shift = 64 - std::countr_zero(_capacity);
  }

  [[nodiscard]] Element *begin() {
    return _dense.get();
  }

  [[nodiscard]] Element *end() {
    return _dense.get() + _size;
  }

  [[nodiscard]] const Element *begin() const {
    return _dense.get();
  }

  [[nodiscard]] const Element *end() const {
    return _dense.get() + _size;
  }

  [[nodiscard]] FixedCapacityRatingMap &entries() {
    return *this;
  }

  [[nodiscard]] const FixedCapacityRatingMap &entries() const {
    return *this;
  }

  [[nodiscard]] bool contains(const Key key) const {
    return find(key).found;
  }

  [[nodiscard]] const Value &get(const Key key) const {
    const ProbeResult result = find(key);
    KASSERT(result.found, "key not in fixed-capacity rating map");
    return _dense[result.dense_index].value;
  }

  Value &operator[](const Key key) {
    const ProbeResult result = find(key);
    if (result.found) {
      return _dense[result.dense_index].value;
    }
    return add_element(key, result.slot)->value;
  }

  void clear() {
    _size = 0;
    if (++_timestamp == 0) [[unlikely]] {
      std::memset(_sparse.get(), 0, MaxCapacity * sizeof(SparseElement));
      _timestamp = 1;
    }
  }

private:
  void allocate() {
    if (!is_allocated()) {
      _sparse = std::make_unique<SparseElement[]>(MaxCapacity);
      _dense.reset(new Element[MaxCapacity]);
      _capacity = MaxCapacity;
      _hash_shift = 64 - std::countr_zero(MaxCapacity);
    }
  }

  [[nodiscard]] ProbeResult find(const Key key) const {
    return find(key, _dense.get());
  }

  [[nodiscard]] ProbeResult find(const Key key, const Element *dense) const {
    std::size_t slot = hash(key);
    const std::size_t start_slot = slot;
    while (true) {
      const std::uint64_t metadata = _sparse[slot].metadata;
      if (sparse_timestamp(metadata) != _timestamp) {
        return {.slot = &_sparse[slot], .dense_index = 0, .found = false};
      }
      const std::uint32_t dense_index = sparse_dense_index(metadata);
      KASSERT(dense_index < _size);
      if (dense[dense_index].key == key) {
        return {.slot = &_sparse[slot], .dense_index = dense_index, .found = true};
      }
      slot = (slot + 1) & (_capacity - 1);
      if (slot == start_slot) {
        return {.slot = nullptr, .dense_index = 0, .found = false};
      }
    }
  }

  [[nodiscard]] Element *add_element(const Key key, SparseElement *sparse) {
    return add_element(key, sparse, _dense.get());
  }

  [[nodiscard]] Element *add_element(const Key key, SparseElement *sparse, Element *dense) {
    KASSERT(sparse != nullptr, "fixed-capacity rating map is full");
    KASSERT(_size < _capacity, "fixed-capacity rating map is full");
    const auto dense_index = static_cast<std::uint32_t>(_size++);
    dense[dense_index] = {.key = key, .value = _initial_value};
    sparse->metadata = pack_sparse_metadata(dense_index, _timestamp);
    return &dense[dense_index];
  }

  Value _initial_value;
  std::unique_ptr<SparseElement[]> _sparse;
  std::unique_ptr<Element[]> _dense;
  std::size_t _capacity = 0;
  std::size_t _size = 0;
  int _hash_shift = 0;
  std::uint32_t _timestamp = 1;
};

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
