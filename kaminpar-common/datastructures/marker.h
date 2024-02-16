/*******************************************************************************
 * @file:   marker.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Marker array with static size. Markets can be reset in amortized
 * constant time.
 ******************************************************************************/
#pragma once

#include <array>
#include <bitset>
#include <limits>
#include <type_traits>

#include "kaminpar-common/assert.h"

namespace kaminpar {
template <std::size_t num_concurrent_markers = 1, typename element_type = std::size_t>
class Marker {
public:
  explicit Marker(const std::size_t capacity)
      : _data(capacity),
        _marker_id(0),
        _first_unmarked_element{0} {}

  Marker(const Marker &) = delete;
  Marker &operator=(const Marker &) = delete;

  Marker(Marker &&) noexcept = default;
  Marker &operator=(Marker &&) noexcept = default;

  template <bool track_first_unmarked_element = false>
  void set(const std::size_t element, const std::size_t marker = 0) {
    KASSERT(marker < num_concurrent_markers);
    KASSERT(element < _data.size());

    _data[element] = ((_data[element] & ~((1u << num_concurrent_markers) - 1u)) == _marker_id)
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
    KASSERT(marker < num_concurrent_markers);
    return _first_unmarked_element[marker];
  }

  [[nodiscard]] bool get(const std::size_t element, const std::size_t marker = 0) const {
    KASSERT(marker < num_concurrent_markers);
    return ((_data[element] & ~((1u << num_concurrent_markers) - 1u)) == _marker_id) &&
           ((_data[element] & (1u << marker)) != 0);
  }

  [[nodiscard]] inline std::size_t size() const {
    return _data.size();
  }
  [[nodiscard]] inline std::size_t capacity() const {
    return size();
  }

  void reset() { // increase such that the least significant
                 // num_concurrent_markers bits are zeroed
    _marker_id |= (1u << num_concurrent_markers) - 1u;
    _marker_id += 1u;
    _first_unmarked_element.fill(0);

    if ((_marker_id | ((1u << num_concurrent_markers) - 1u)) ==
        std::numeric_limits<element_type>::max()) {
      _marker_id = 0;
      const auto capacity = _data.size();
      _data.clear();
      _data.resize(capacity, 0);
    }
  }

  void resize(const std::size_t capacity) {
    _data.resize(capacity);
  }

  [[nodiscard]] std::size_t memory_in_kb() const {
    return _data.size() * sizeof(element_type) / 1000;
  }

private:
  std::vector<element_type> _data;
  element_type _marker_id;
  std::array<std::size_t, num_concurrent_markers> _first_unmarked_element;
};
} // namespace kaminpar
