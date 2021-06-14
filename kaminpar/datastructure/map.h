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
 * Briggs, Preston, and Linda Torczon. "An efficient representation for sparse sets."
 * ACM Letters on Programming Languages and Systems (LOPLAS) 2.1-4 (1993): 59-69.
 */
#pragma once

#include "utility/math.h"

namespace kaminpar {
template<typename Key, typename Value>
struct MapElement {
  Key key;
  Value value;
};

/*!
 * Sparse map implementation that uses a fixed size.
 * In contrast to the implementation in KaHyPar (see kahypar/datastructure/sparse_map.h),
 * which uses as size the cardinality of the key universe, hash collisions have to be handled
 * explicitly. Hash collisions are resolved with linear probing.
 * Advantage of the implementation is that it uses significantly less space than the
 * version in KaHyPar and should be therefore more cache-efficient.
 * Note, there is no fallback strategy if all slots of the sparse map are occupied by an
 * element. Please make sure that no more than MAP_SIZE elements are inserted into the
 * sparse map. Otherwise, the behavior is undefined.
 */
template<typename Key, typename Value, std::size_t fixed_size = 32768> // Size of sparse map is approx. 1 MB
class FixedSizeSparseMap {
  using Element = MapElement<Key, Value>;

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

  FixedSizeSparseMap(FixedSizeSparseMap &&other) noexcept
      : _map_size(other._map_size),
        _initial_value(other._initial_value),
        _data(std::move(other._data)),
        _size(other._size),
        _timestamp(other._timestamp),
        _sparse(std::move(other._sparse)),
        _dense(std::move(other._dense)) {
    other._data = nullptr;
    other._sparse = nullptr;
    other._dense = nullptr;
  }

  // Query functions
  [[nodiscard]] std::size_t capacity() const { return _map_size; }
  [[nodiscard]] std::size_t size() const { return _size; }
  [[nodiscard]] const Element *begin() const { return _dense; }
  [[nodiscard]] const Element *end() const { return _dense + _size; }
  [[nodiscard]] Element *begin() { return _dense; }
  [[nodiscard]] Element *end() { return _dense + _size; }
  [[nodiscard]] bool contains(const Key key) const { return containsValidElement(key, find(key)); }
  [[nodiscard]] const Value &get(const Key key) const { return find(key)->element->value; }

  void set_max_size(const std::size_t max_size) {
    if (max_size > _map_size) {
      freeInternalData();
      allocate(max_size);
    }
  }

  Value &operator[](const Key key) {
    SparseElement *s = find(key);
    if (containsValidElement(key, s)) {
      ASSERT(s->element);
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
    ASSERT(_size < _map_size);
    std::size_t hash = key & (_map_size - 1);
    while (_sparse[hash].timestamp == _timestamp) {
      ASSERT(_sparse[hash].element);
      if (_sparse[hash].element->key == key) { return &_sparse[hash]; }
      hash = (hash + 1) & (_map_size - 1);
    }
    return &_sparse[hash];
  }

  inline bool containsValidElement(const Key key, const SparseElement *s) const {
    ASSERT(s);
    const bool is_contained = s->timestamp == _timestamp;
    ASSERT(!is_contained || s->element->key == key);
#ifndef KAMINPAR_ENABLE_ASSERTIONS
    (void) key;
#endif
    return is_contained;
  }

  inline Element *addElement(const Key key, const Value value, SparseElement *s) {
    ASSERT(find(key) == s);
    _dense[_size] = Element{key, value};
    *s = SparseElement{&_dense[_size++], _timestamp};
    return s->element;
  }

  void allocate(const std::size_t size) {
    if (_data == nullptr) {
      _map_size = align_to_next_power_of_two(size);
      _data = std::make_unique<uint8_t[]>(_map_size * sizeof(Element) + _map_size * sizeof(SparseElement));
      _size = 0;
      _timestamp = 1;
      _sparse = reinterpret_cast<SparseElement *>(_data.get());
      _dense = reinterpret_cast<Element *>(_data.get() + +sizeof(SparseElement) * _map_size);
      memset(_data.get(), 0, _map_size * (sizeof(Element) + sizeof(SparseElement)));
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

template<typename Key, typename Value>
class SparseMap {
  using Element = MapElement<Key, Value>;

public:
  SparseMap() : _capacity{0} {}
  explicit SparseMap(const std::size_t capacity) : _capacity{capacity} { allocate_data(capacity); }

  SparseMap(const SparseMap &) = delete;
  SparseMap &operator=(const SparseMap &) = delete;
  SparseMap(SparseMap &&) noexcept = default;
  SparseMap &operator=(SparseMap &&) noexcept = default;

  std::size_t capacity() const { return _capacity; }
  bool is_allocated() const { return capacity() > 0; }
  std::size_t size() const { return _size; }

  void shrink(const std::size_t capacity) { _dense = reinterpret_cast<Element *>(_sparse + capacity); }
  void resize(const std::size_t capacity) { allocate_data(capacity); }

  bool contains(const Key key) const {
    const std::size_t index{_sparse[key]};
    return index < _size && _dense[index].key == key;
  }

  void add(const Key key, const Value value) {
    if (!contains(key)) {
      _dense[_size] = {key, value};
      _sparse[key] = _size++;
    }
  }

  void remove(const Key key) {
    const std::size_t index{_sparse[key]};
    if (index < _size && _dense[index].key == key) {
      std::swap(_dense[index], _dense[_size - 1]);
      _sparse[_dense[index].key] = index;
      --_size;
    }
  }

  const Element *begin() const { return _dense; }
  const Element *end() const { return _dense + _size; }
  Element *begin() { return _dense; }
  Element *end() { return _dense + _size; }

  void clear() { _size = 0; }

  Value &operator[](const Key key) {
    if (!contains(key)) {
      _dense[_size] = Element{key, Value()};
      _sparse[key] = _size++;
    }

    return _dense[_sparse[key]].value;
  }

  const Value &get(const Key key) const {
    ASSERT(contains(key)) << "key not in sparse map: " << key;
    return _dense[_sparse[key]].value;
  }

private:
  void allocate_data(const std::size_t capacity) {
    _capacity = capacity;

    ASSERT(!_data);
    const std::size_t total_memory{_capacity * sizeof(Element) + _capacity * sizeof(std::size_t)};
    const std::size_t num_elements{static_cast<std::size_t>(std::ceil(1.0 * total_memory / sizeof(std::size_t)))};
    _data = parallel::make_unique<std::size_t>(num_elements);
    _sparse = reinterpret_cast<std::size_t *>(_data.get());
    _dense = reinterpret_cast<Element *>(_sparse + _capacity);
  }

  std::size_t _capacity{};
  std::size_t _size{0};
  parallel::tbb_unique_ptr<std::size_t> _data{nullptr};
  std::size_t *_sparse{nullptr};
  Element *_dense{nullptr};
};

template<typename Value>
class RatingMap {
  using SmallMap = FixedSizeSparseMap<NodeID, Value>;
  using LargeMap = SparseMap<NodeID, Value>;
  using Element = MapElement<NodeID, Value>;

public:
  enum class MapType { SMALL, LARGE };

  RatingMap(const std::size_t max_size) : _max_size{max_size} {}

  MapType update_upper_bound_size(const std::size_t upper_bound_size) {
    select_map(upper_bound_size);
    return _selected_map;
  }

  Value &operator[](const NodeID u) {
    switch (_selected_map) {
      case MapType::SMALL: return _small_map[u];
      case MapType::LARGE: return _large_map[u];
    }
    __builtin_unreachable();
  }

  const Element *begin() const {
    switch (_selected_map) {
      case MapType::SMALL: return _small_map.begin();
      case MapType::LARGE: return _large_map.begin();
    }
    __builtin_unreachable();
  }

  const Element *end() const {
    switch (_selected_map) {
      case MapType::SMALL: return _small_map.end();
      case MapType::LARGE: return _large_map.end();
    }
    __builtin_unreachable();
  }

  Element *begin() { return const_cast<Element *>(static_cast<const RatingMap<Value> *>(this)->begin()); }
  Element *end() { return const_cast<Element *>(static_cast<const RatingMap<Value> *>(this)->end()); }

  void clear() {
    switch (_selected_map) {
      case MapType::SMALL: _small_map.clear(); break;
      case MapType::LARGE: _large_map.clear(); break;
    }
  }

  bool contains(const NodeID key) {
    switch (_selected_map) {
      case MapType::SMALL: return _small_map.contains(key);
      case MapType::LARGE: return _large_map.contains(key);
    }
    __builtin_unreachable();
  }

  std::size_t small_map_counter() const { return _small_map_counter; }
  std::size_t large_map_counter() const { return _large_map_counter; }

  std::size_t size() const {
    switch (_selected_map) {
      case MapType::SMALL: return _small_map.size();
      case MapType::LARGE: return _large_map.size();
    }
    __builtin_unreachable();
  }

private:
  void select_map(const std::size_t upper_bound_size) {
    if (_max_size < SmallMap::MAP_SIZE || upper_bound_size > SmallMap::MAP_SIZE / 3) {
      _selected_map = MapType::LARGE;
      ++_large_map_counter;
    } else {
      _selected_map = MapType::SMALL;
      ++_small_map_counter;
    }

    if (_selected_map == MapType::LARGE && !_large_map.is_allocated()) {
      _large_map.resize(_max_size);
      ASSERT(_large_map.is_allocated());
    }
  }

  std::size_t _max_size;
  MapType _selected_map{MapType::SMALL};
  SmallMap _small_map{};
  LargeMap _large_map{}; // allocate on demand

  std::size_t _small_map_counter{0};
  std::size_t _large_map_counter{0};
};

template<typename Key, typename Value>
class DynamicSparseMap {
  struct MapElement {
    Key key;
    Value value;
  };

  struct SparseElement {
    MapElement *element;
    size_t timestamp;
  };

public:
  static constexpr size_t MAP_SIZE = 32768; // Size of sparse map is approx. 1 MB

  static_assert(MAP_SIZE && ((MAP_SIZE & (MAP_SIZE - 1)) == 0UL), "Size of map is not a power of two!");

  explicit DynamicSparseMap()
      : _capacity(0),
        _initial_value(),
        _data(nullptr),
        _size(0),
        _timestamp(1),
        _sparse(nullptr),
        _dense(nullptr) {
    allocate(MAP_SIZE);
  }

  DynamicSparseMap(const DynamicSparseMap &) = delete;
  DynamicSparseMap &operator=(const DynamicSparseMap &other) = delete;

  DynamicSparseMap(DynamicSparseMap &&other)
      : _capacity(other._capacity),
        _initial_value(other._initial_value),
        _data(std::move(other._data)),
        _size(other._size),
        _timestamp(other._timestamp),
        _sparse(std::move(other._sparse)),
        _dense(std::move(other._dense)) {
    other._data = nullptr;
    other._sparse = nullptr;
    other._dense = nullptr;
  }

  ~DynamicSparseMap() = default;

  size_t capacity() const { return _capacity; }

  size_t size() const { return _size; }

  const MapElement *begin() const { return _dense; }

  const MapElement *end() const { return _dense + _size; }

  MapElement *begin() { return _dense; }

  MapElement *end() { return _dense + _size; }

  bool contains(const Key key) const {
    SparseElement *s = find(key, _sparse, _capacity);
    return containsValidElement(key, s);
  }

  Value &operator[](const Key key) {
    SparseElement *s = find(key, _sparse, _capacity);
    if (containsValidElement(key, s)) {
      ASSERT(s->element);
      return s->element->value;
    } else {
      if (_size + 1 > _capacity / 5UL) {
        grow();
        s = find(key, _sparse, _capacity);
      }
      return addElement(key, _initial_value, s, _dense, _size)->value;
    }
  }

  const Value &get(const Key key) const {
    ASSERT(contains(key));
    return find(key, _sparse, _capacity)->element->value;
  }

  const Value *get_if_contained(const Key key) const {
    SparseElement *s = find(key, _sparse, _capacity);
    if (containsValidElement(key, s)) {
      return &s->element->value;
    } else {
      return nullptr;
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
  inline SparseElement *find(const Key key, SparseElement *sparse, const size_t capacity) const {
    size_t hash = key & (capacity - 1);
    while (sparse[hash].timestamp == _timestamp) {
      ASSERT(sparse[hash].element);
      if (sparse[hash].element->key == key) { return &sparse[hash]; }
      hash = (hash + 1) & (capacity - 1);
    }
    return &sparse[hash];
  }

  inline bool containsValidElement(const Key key, const SparseElement *s) const {
    (void) key;
    ASSERT(s);
    const bool is_contained = s->timestamp == _timestamp;
    ASSERT(!is_contained || s->element->key == key);
    return is_contained;
  }

  inline MapElement *addElement(const Key key, const Value value, SparseElement *s, MapElement *dense, size_t &size) {
    dense[size] = MapElement{key, value};
    *s = SparseElement{&dense[size++], _timestamp};
    return s->element;
  }

  void allocate(const size_t size) {
    if (_data == nullptr) {
      _capacity = align_to_next_power_of_two(size);
      _data = std::make_unique<uint8_t[]>(_capacity * sizeof(MapElement) + _capacity * sizeof(SparseElement));
      _size = 0;
      _timestamp = 1;
      _sparse = reinterpret_cast<SparseElement *>(_data.get());
      _dense = reinterpret_cast<MapElement *>(_data.get() + sizeof(SparseElement) * _capacity);
      memset(_data.get(), 0, _capacity * (sizeof(MapElement) + sizeof(SparseElement)));
    }
  }

  void grow() {
    const size_t capacity = 2UL * _capacity;
    std::unique_ptr<uint8_t[]> data = std::make_unique<uint8_t[]>(capacity * sizeof(MapElement) +
                                                                  capacity * sizeof(SparseElement));
    SparseElement *sparse = reinterpret_cast<SparseElement *>(data.get());
    MapElement *dense = reinterpret_cast<MapElement *>(data.get() + sizeof(SparseElement) * capacity);
    memset(data.get(), 0, capacity * (sizeof(MapElement) + sizeof(SparseElement)));

    rehash(sparse, dense, capacity);

    _data = std::move(data);
    _sparse = sparse;
    _dense = dense;
    _capacity = capacity;
  }

  void rehash(SparseElement *sparse, MapElement *dense, const size_t capacity) {
    size_t size = 0;
    for (const MapElement &element : *this) {
      SparseElement *slot = find(element.key, sparse, capacity);
      addElement(element.key, element.value, slot, dense, size);
    }
    ASSERT(size == _size);
  }

  size_t align_to_next_power_of_two(const size_t size) const {
    return std::pow(2.0, std::ceil(std::log2(static_cast<double>(size))));
  }

  size_t _capacity;
  const Value _initial_value;
  std::unique_ptr<uint8_t[]> _data;

  size_t _size;
  size_t _timestamp;
  SparseElement *_sparse;
  MapElement *_dense;
};
} // namespace kaminpar