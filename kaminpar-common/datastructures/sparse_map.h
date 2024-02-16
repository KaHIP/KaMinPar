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

#include <cmath>
#include <cstdlib>
#include <memory>

#include "kaminpar-common/assert.h"

namespace kaminpar {
template <typename Key, typename Value> class SparseMap {
  struct Element {
    Key key;
    Value value;
  };

public:
  SparseMap() = default;

  explicit SparseMap(const std::size_t capacity) : _capacity(capacity) {
    allocate_data(capacity);
  }

  SparseMap(const SparseMap &) = delete;
  SparseMap &operator=(const SparseMap &) = delete;

  SparseMap(SparseMap &&) noexcept = default;
  SparseMap &operator=(SparseMap &&) noexcept = default;

  [[nodiscard]] std::size_t capacity() const {
    return _capacity;
  }

  [[nodiscard]] bool is_allocated() const {
    return capacity() > 0;
  }

  [[nodiscard]] std::size_t size() const {
    return _size;
  }

  void shrink(const std::size_t capacity) {
    _dense = reinterpret_cast<Element *>(_sparse + capacity);
  }

  void resize(const std::size_t capacity) {
    allocate_data(capacity);
  }

  bool contains(const Key key) const {
    KASSERT(_data != nullptr);
    KASSERT(key < _capacity);

    const std::size_t index = _sparse[key];
    return index < _size && _dense[index].key == key;
  }

  void add(const Key key, const Value value) {
    if (!contains(key)) {
      _dense[_size] = {key, value};
      _sparse[key] = _size++;
    }
  }

  void remove(const Key key) {
    const std::size_t index = _sparse[key];
    if (index < _size && _dense[index].key == key) {
      std::swap(_dense[index], _dense[_size - 1]);
      _sparse[_dense[index].key] = index;
      --_size;
    }
  }

  const Element *begin() const {
    return _dense;
  }

  const Element *end() const {
    return _dense + _size;
  }

  Element *begin() {
    return _dense;
  }

  Element *end() {
    return _dense + _size;
  }

  auto &entries() {
    return *this;
  }

  void clear() {
    _size = 0;
  }

  Value &operator[](const Key key) {
    if (!contains(key)) {
      _dense[_size] = Element{key, Value()};
      _sparse[key] = _size++;
    }

    return _dense[_sparse[key]].value;
  }

  const Value &get(const Key key) const {
    KASSERT(contains(key), "key not in sparse map: " << key);
    return _dense[_sparse[key]].value;
  }

private:
  void allocate_data(const std::size_t capacity) {
    _capacity = capacity;

    KASSERT(!_data);
    const std::size_t total_memory = _capacity * sizeof(Element) + _capacity * sizeof(std::size_t);
    const std::size_t num_elements = std::ceil(1.0 * total_memory / sizeof(std::size_t));
    _data = std::make_unique<std::size_t[]>(num_elements);
    _sparse = reinterpret_cast<std::size_t *>(_data.get());
    _dense = reinterpret_cast<Element *>(_sparse + _capacity);
  }

  std::size_t _capacity = 0;
  std::size_t _size = 0;
  std::unique_ptr<std::size_t[]> _data = nullptr;
  std::size_t *_sparse = nullptr;
  Element *_dense = nullptr;
};
} // namespace kaminpar
