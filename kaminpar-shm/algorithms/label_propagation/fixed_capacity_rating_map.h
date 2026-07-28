/*******************************************************************************
 * Cache-efficient fixed-capacity map for neighborhood rating aggregation.
 *
 * @file:   fixed_capacity_rating_map.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/math.h"

namespace kaminpar::shm::lp {

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

} // namespace kaminpar::shm::lp
