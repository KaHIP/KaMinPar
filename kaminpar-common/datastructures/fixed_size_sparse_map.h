/*******************************************************************************
 * This file is part of KaHyPar.
 *
 * Copyright (C) 2019 Tobias Heuer <tobias.heuer@kit.edu>
 * Copyright (C) 2016 Sebastian Schlag <sebastian.schlag@kit.edu>
 *
 * KaHyPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaHyPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaHyPar.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
/*
 * Sparse map based on sparse set representation of
 * Briggs, Preston, and Linda Torczon. "An efficient representation for sparse
 * sets." ACM Letters on Programming Languages and Systems (LOPLAS) 2.1-4
 * (1993): 59-69.
 */
#pragma once

#include <cstring>
#include <memory>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/math.h"

namespace kaminpar {
/*!
 * Sparse map implementation that uses a fixed size.
 * In contrast to the implementation in KaHyPar (see
 * kahypar/datastructure/sparse_map.h), which uses as size the cardinality of
 * the key universe, hash collisions have to be handled explicitly. Hash
 * collisions are resolved with linear probing. Advantage of the implementation
 * is that it uses significantly less space than the version in KaHyPar and
 * should be therefore more cache-efficient. Note, there is no fallback strategy
 * if all slots of the sparse map are occupied by an element. Please make sure
 * that no more than MAP_SIZE elements are inserted into the sparse map.
 * Otherwise, the behavior is undefined.
 */
template <
    typename Key,
    typename Value,
    std::size_t fixed_size = 32768> // Size of sparse map is approx. 1 MB
class FixedSizeSparseMap {
  struct Element {
    Key key;
    Value value;
  };

  struct SparseElement {
    Element *element;
    std::size_t timestamp;
  };

public:
  static constexpr std::size_t MAP_SIZE = fixed_size;
  static_assert(math::is_power_of_2(MAP_SIZE), "Size of map is not a power of two!");

  explicit FixedSizeSparseMap(const Value initial_value = Value())
      : _map_size(0),
        _initial_value(initial_value),
        _data(nullptr),
        _size(0),
        _timestamp(1),
        _sparse(nullptr),
        _dense(nullptr) {
    allocate(MAP_SIZE);
  }

  explicit FixedSizeSparseMap(const std::size_t max_size, const Value initial_value = Value())
      : _map_size(0),
        _initial_value(initial_value),
        _data(nullptr),
        _size(0),
        _timestamp(1),
        _sparse(nullptr),
        _dense(nullptr) {
    allocate(max_size);
  }

  FixedSizeSparseMap(const FixedSizeSparseMap &) = delete;
  FixedSizeSparseMap &operator=(const FixedSizeSparseMap &other) = delete;

  // Query functions
  [[nodiscard]] std::size_t capacity() const {
    return _map_size;
  }
  [[nodiscard]] std::size_t size() const {
    return _size;
  }
  [[nodiscard]] const Element *begin() const {
    return _dense;
  }
  [[nodiscard]] const Element *end() const {
    return _dense + _size;
  }
  [[nodiscard]] Element *begin() {
    return _dense;
  }
  [[nodiscard]] Element *end() {
    return _dense + _size;
  }
  [[nodiscard]] bool contains(const Key key) const {
    return containsValidElement(key, find(key));
  }
  [[nodiscard]] const Value &get(const Key key) const {
    return find(key)->element->value;
  }

  auto &entries() {
    return *this;
  }

  void set_max_size(const std::size_t max_size) {
    if (max_size > _map_size) {
      freeInternalData();
      allocate(max_size);
    }
  }

  Value &operator[](const Key key) {
    SparseElement *s = find(key);
    if (containsValidElement(key, s)) {
      KASSERT(s->element);
      return s->element->value;
    } else {
      return addElement(key, _initial_value, s)->value;
    }
  }

  void clear() {
    _size = 0;
    ++_timestamp;
  }

  void freeInternalData() {
    _size = 0;
    _timestamp = 0;
    _data = nullptr;
    _sparse = nullptr;
    _dense = nullptr;
  }

private:
  inline SparseElement *find(const Key key) const {
    KASSERT(_size < _map_size);
    std::size_t hash = key & (_map_size - 1);
    while (_sparse[hash].timestamp == _timestamp) {
      KASSERT(_sparse[hash].element);
      if (_sparse[hash].element->key == key) {
        return &_sparse[hash];
      }
      hash = (hash + 1) & (_map_size - 1);
    }
    return &_sparse[hash];
  }

  inline bool containsValidElement([[maybe_unused]] const Key key, const SparseElement *s) const {
    KASSERT(s);
    const bool is_contained = s->timestamp == _timestamp;
    KASSERT((!is_contained || s->element->key == key));
    return is_contained;
  }

  inline Element *addElement(const Key key, const Value value, SparseElement *s) {
    KASSERT(find(key) == s);
    _dense[_size] = Element{key, value};
    *s = SparseElement{&_dense[_size++], _timestamp};
    return s->element;
  }

  void allocate(const std::size_t size) {
    if (_data == nullptr) {
      _map_size = align_to_next_power_of_two(size);
      _data = std::make_unique<uint8_t[]>(
          _map_size * sizeof(Element) + _map_size * sizeof(SparseElement)
      );
      _size = 0;
      _timestamp = 1;
      _sparse = reinterpret_cast<SparseElement *>(_data.get());
      _dense = reinterpret_cast<Element *>(_data.get() + +sizeof(SparseElement) * _map_size);
      std::memset(_data.get(), 0, _map_size * (sizeof(Element) + sizeof(SparseElement)));
    }
  }

  [[nodiscard]] std::size_t align_to_next_power_of_two(const std::size_t size) const {
    return std::pow(2.0, std::ceil(std::log2(static_cast<double>(size))));
  }

  std::size_t _map_size;
  const Value _initial_value;
  std::unique_ptr<uint8_t[]> _data;

  std::size_t _size;
  std::size_t _timestamp;
  SparseElement *_sparse;
  Element *_dense;
};
} // namespace kaminpar
