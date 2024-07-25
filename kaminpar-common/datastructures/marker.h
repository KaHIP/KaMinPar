/*******************************************************************************
 * Sequential timestamp marker data structure of static size. Markers can be
 * reset in amortized constant time.
 *
 * @file:   marker.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <array>
#include <limits>
#include <vector>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/heap_profiler.h"

namespace kaminpar {
template <std::size_t kNumConcurrentMarkers = 1, typename Value = std::size_t> class Marker {
public:
  explicit Marker() : _marker_id(0), _first_unmarked_element{0} {
    RECORD_DATA_STRUCT(0, _struct);
  }

  explicit Marker(const std::size_t capacity)
      : _data(capacity),
        _marker_id(0),
        _first_unmarked_element{0} {
    RECORD_DATA_STRUCT(capacity * sizeof(Value), _struct);
  }

  Marker(const Marker &) = delete;
  Marker &operator=(const Marker &) = delete;

  Marker(Marker &&) noexcept = default;
  Marker &operator=(Marker &&) noexcept = default;

  template <bool track_first_unmarked_element = false>
  void set(const std::size_t element, const std::size_t marker = 0) {
    KASSERT(marker < kNumConcurrentMarkers);
    KASSERT(element < _data.size());

    _data[element] = ((_data[element] & ~((1u << kNumConcurrentMarkers) - 1u)) == _marker_id)
                         ? _data[element] | (1u << marker)
                         : _marker_id | (1u << marker);

    if constexpr (track_first_unmarked_element) {
      while (_first_unmarked_element[marker] < _data.size() &&
             get(_first_unmarked_element[marker], marker)) {
        ++_first_unmarked_element[marker];
      }
    }
  }

  [[nodiscard]] inline std::size_t first_unmarked_element(const std::size_t marker = 0) const {
    KASSERT(marker < kNumConcurrentMarkers);
    return _first_unmarked_element[marker];
  }

  [[nodiscard]] bool get(const std::size_t element, const std::size_t marker = 0) const {
    KASSERT(marker < kNumConcurrentMarkers);
    return ((_data[element] & ~((1u << kNumConcurrentMarkers) - 1u)) == _marker_id) &&
           ((_data[element] & (1u << marker)) != 0);
  }

  [[nodiscard]] inline std::size_t size() const {
    return _data.size();
  }

  [[nodiscard]] inline std::size_t capacity() const {
    return size();
  }

  // Increase timestamp s.t. the least significant
  // num_concurrent_markers bits are zeroed
  void reset() {
    _marker_id |= (1u << kNumConcurrentMarkers) - 1u;
    _marker_id += 1u;
    _first_unmarked_element.fill(0);

    if ((_marker_id | ((1u << kNumConcurrentMarkers) - 1u)) == std::numeric_limits<Value>::max()) {
      _marker_id = 0;
      const auto capacity = _data.size();
      _data.clear();
      _data.resize(capacity, 0);
    }
  }

  void resize(const std::size_t capacity) {
    IF_HEAP_PROFILING(_struct->size = std::max(_struct->size, capacity * sizeof(Value)));
    _data.resize(capacity);
  }

  [[nodiscard]] std::size_t memory_in_kb() const {
    return _data.size() * sizeof(Value) / 1000;
  }

private:
  std::vector<Value> _data;
  Value _marker_id;
  std::array<std::size_t, kNumConcurrentMarkers> _first_unmarked_element;

  IF_HEAP_PROFILING(heap_profiler::DataStructure *_struct);
};
} // namespace kaminpar
