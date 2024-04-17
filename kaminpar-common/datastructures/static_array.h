/*******************************************************************************
 * @file:   static_array.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Combination of owning static array and a span.
 ******************************************************************************/
#pragma once

#include <cstring>
#include <initializer_list>
#include <iterator>
#include <thread>
#include <vector>

#include <tbb/parallel_for.h>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/parallel/tbb_malloc.h"

namespace kaminpar {
namespace static_array {
constexpr struct noinit_t {
} noinit;
} // namespace static_array

template <typename T> class StaticArray {
public:
  class StaticArrayIterator {
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using reference = T &;
    using pointer = T *;
    using difference_type = std::ptrdiff_t;

    StaticArrayIterator() : _ptr(nullptr) {}
    StaticArrayIterator(T *ptr) : _ptr(ptr) {}

    StaticArrayIterator(const StaticArrayIterator &other) = default;
    StaticArrayIterator &operator=(const StaticArrayIterator &other) = default;

    reference operator*() const {
      return *_ptr;
    }

    pointer operator->() const {
      return _ptr;
    }

    StaticArrayIterator &operator++() {
      return ++_ptr, *this;
    }

    StaticArrayIterator &operator--() {
      return --_ptr, *this;
    }

    StaticArrayIterator operator++(int) {
      return {_ptr++};
    }

    StaticArrayIterator operator--(int) {
      return {_ptr--};
    }

    StaticArrayIterator operator+(const difference_type &n) const {
      return StaticArrayIterator{_ptr + n};
    }

    StaticArrayIterator &operator+=(const difference_type &n) {
      return _ptr += n, *this;
    }

    StaticArrayIterator operator-(const difference_type &n) const {
      return StaticArrayIterator{_ptr - n};
    }

    StaticArrayIterator &operator-=(const difference_type &n) {
      return _ptr -= n, *this;
    }

    reference operator[](const difference_type &n) const {
      return *_ptr[n];
    }

    bool operator==(const StaticArrayIterator &other) const {
      return _ptr == other._ptr;
    }

    bool operator!=(const StaticArrayIterator &other) const {
      return _ptr != other._ptr;
    }

    bool operator>(const StaticArrayIterator &other) const {
      return _ptr > other._ptr;
    }

    bool operator<(const StaticArrayIterator &other) const {
      return _ptr < other._ptr;
    }

    bool operator<=(const StaticArrayIterator &other) const {
      return _ptr <= other._ptr;
    }

    bool operator>=(const StaticArrayIterator &other) const {
      return _ptr >= other._ptr;
    }

    difference_type operator+(const StaticArrayIterator &other) {
      return _ptr + other._ptr;
    }

    difference_type operator-(const StaticArrayIterator &other) {
      return _ptr - other._ptr;
    }

  private:
    T *_ptr;
  };

public:
  using value_type = T;
  using size_type = std::size_t;
  using reference = T &;
  using const_reference = const T &;
  using iterator = StaticArrayIterator;
  using const_iterator = const StaticArrayIterator;

  StaticArray(T *storage, const std::size_t size) : _size(size), _data(storage) {
    RECORD_DATA_STRUCT(size * sizeof(T), _struct);
  }

  StaticArray(const std::size_t start, const std::size_t size, StaticArray &data)
      : StaticArray(size, data._data + start) {
    KASSERT(start + size <= data.size());
  }

  StaticArray(const std::size_t size, value_type *data) : _size(size), _data(data) {
    RECORD_DATA_STRUCT(size * sizeof(T), _struct);
  }

  StaticArray(const std::size_t size, const value_type init_value = value_type()) {
    RECORD_DATA_STRUCT(0, _struct);
    resize(size, init_value);
  }

  StaticArray(const std::size_t size, static_array::noinit_t) {
    RECORD_DATA_STRUCT(0, _struct);
    resize(size, static_array::noinit);
  }

  template <typename Iterator>
  StaticArray(Iterator first, Iterator last) : StaticArray(std::distance(first, last)) {
    tbb::parallel_for<std::size_t>(0, _size, [&](const std::size_t i) { _data[i] = *(first + i); });
  }

  StaticArray() : StaticArray(0) {}

  StaticArray(const StaticArray &) = delete;
  StaticArray &operator=(const StaticArray &) = delete;

  StaticArray(StaticArray &&) noexcept = default;
  StaticArray &operator=(StaticArray &&) noexcept = default;

  bool operator==(const StaticArray<T> &other) const {
    if (size() != other.size()) {
      return false;
    }
    return std::memcmp(_data, other._data, size()) == 0;
  }

  //
  // Data access members
  //

  void write(const size_type pos, const_reference value) {
    at(pos) = value;
  }

  reference at(const size_type pos) {
    return _data[pos];
  }

  const_reference at(const size_type pos) const {
    return _data[pos];
  }

  reference operator[](const size_type pos) {
    KASSERT(pos < _size);
    return _data[pos];
  }

  const_reference operator[](const size_type pos) const {
    return _data[pos];
  }

  reference back() {
    KASSERT(_data);
    KASSERT(_size > 0u);
    return _data[_size - 1];
  }

  const_reference back() const {
    KASSERT(_data);
    KASSERT(_size > 0u);
    return _data[_size - 1];
  }

  reference front() {
    KASSERT(_data);
    KASSERT(_size > 0u);
    return _data[0];
  }

  const_reference front() const {
    KASSERT(_data);
    KASSERT(_size > 0u);
    return _data[0];
  }

  value_type *data() {
    KASSERT(_data);
    return _data;
  }

  const value_type *data() const {
    KASSERT(_data);
    return _data;
  }

  //
  // Iterators
  //

  iterator begin() {
    KASSERT(_data);
    return iterator(_data);
  }

  const_iterator cbegin() const {
    KASSERT(_data);
    return const_iterator(_data);
  }

  iterator end() {
    return iterator{_data + _size};
  }

  const_iterator cend() const {
    return const_iterator{_data + _size};
  }

  const_iterator begin() const {
    return cbegin();
  }
  iterator end() const {
    return cend();
  }

  void restrict(const std::size_t new_size) {
    KASSERT(
        new_size <= _size,
        "restricted size " << new_size << " must be smaller than the unrestricted size " << _size
    );
    _unrestricted_size = _size;
    _size = new_size;
  }

  void unrestrict() {
    _size = _unrestricted_size;
  }

  //
  // Capacity
  //

  [[nodiscard]] bool empty() const {
    return _size == 0;
  }
  [[nodiscard]] size_type size() const {
    return _size;
  }

  void resize(const std::size_t size, static_array::noinit_t) {
    KASSERT(_data == _owned_data.get(), "cannot resize span", assert::always);
    allocate_data(size);
  }

  void resize(
      const size_type size,
      const value_type init_value = value_type(),
      const bool assign_parallel = true
  ) {
    resize(size, static_array::noinit);
    assign(size, init_value, assign_parallel);
  }

  void assign(const size_type count, const value_type value, const bool assign_parallel = true) {
    KASSERT(_data);

    if (assign_parallel) {
      const std::size_t step{std::max(count / std::thread::hardware_concurrency(), 1UL)};
      tbb::parallel_for(0UL, count, step, [&](const size_type i) {
        for (size_type j = i; j < std::min(i + step, count); ++j) {
          _data[j] = value;
        }
      });
    } else {
      for (std::size_t i = 0; i < count; ++i) {
        _data[i] = value;
      }
    }
  }

  parallel::tbb_unique_ptr<value_type> free() {
    _size = 0;
    _unrestricted_size = 0;
    _data = nullptr;
    return std::move(_owned_data);
  }

private:
  void allocate_data(const std::size_t size) {
    _owned_data = parallel::make_unique<value_type>(size);
    _data = _owned_data.get();
    _size = size;
    _unrestricted_size = _size;

    IF_HEAP_PROFILING(_struct->size = std::max(_struct->size, size * sizeof(value_type)));
  }

  size_type _size = 0;
  size_type _unrestricted_size = 0;
  parallel::tbb_unique_ptr<value_type> _owned_data = nullptr;
  value_type *_data = nullptr;

  IF_HEAP_PROFILING(heap_profiler::DataStructure *_struct);
};

namespace static_array {
template <typename T> StaticArray<T> create(std::initializer_list<T> list) {
  return {list.begin(), list.end()};
}

template <typename T> StaticArray<T> create(const std::vector<T> &vec) {
  return {vec.begin(), vec.end()};
}
} // namespace static_array
} // namespace kaminpar
