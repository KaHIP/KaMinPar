/*******************************************************************************
 * Tiny linear map for neighborhood rating aggregation.
 *
 * @file:   linear_rating_map.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <array>
#include <cstddef>

#include "kaminpar-common/assert.h"

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

} // namespace kaminpar::shm::lp
