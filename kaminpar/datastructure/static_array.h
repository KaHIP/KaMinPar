#pragma once

#include "definitions.h"
#include "parallel.h"

#include <iterator>
#include <tbb/parallel_for.h>
#include <thread>
#include <vector>

namespace kaminpar {
template<typename T>
class StaticArray {
public:
  class StaticArrayIterator : public std::iterator<std::random_access_iterator_tag, T> {
    using Base = std::iterator<std::random_access_iterator_tag, T>;

  public:
    using value_type = typename Base::value_type;
    using reference = typename Base::reference;
    using pointer = typename Base::pointer;
    using difference_type = typename Base::difference_type;

    StaticArrayIterator() : _ptr(nullptr) {}
    explicit StaticArrayIterator(T *ptr) : _ptr(ptr) {}
    StaticArrayIterator(const StaticArrayIterator &other) : _ptr(other._ptr) {}

    reference operator*() const { return *_ptr; }
    pointer operator->() const { return _ptr; }

    StaticArrayIterator &operator++() { return ++_ptr, *this; }
    StaticArrayIterator &operator--() { return --_ptr, *this; }
    StaticArrayIterator operator++(int) { return {_ptr++}; }
    StaticArrayIterator operator--(int) { return {_ptr--}; }
    StaticArrayIterator operator+(const difference_type &n) const { return StaticArrayIterator{_ptr + n}; }
    StaticArrayIterator &operator+=(const difference_type &n) { return _ptr += n, *this; }
    StaticArrayIterator operator-(const difference_type &n) const { return StaticArrayIterator{_ptr - n}; }
    StaticArrayIterator &operator-=(const difference_type &n) { return _ptr -= n, *this; }

    reference operator[](const difference_type &n) const { return *_ptr[n]; }
    bool operator==(const StaticArrayIterator &other) { return _ptr == other._ptr; }
    bool operator!=(const StaticArrayIterator &other) { return _ptr != other._ptr; }
    bool operator>(const StaticArrayIterator &other) const { return _ptr > other._ptr; }
    bool operator<(const StaticArrayIterator &other) const { return _ptr < other._ptr; }
    bool operator<=(const StaticArrayIterator &other) const { return _ptr <= other._ptr; }
    bool operator>=(const StaticArrayIterator &other) const { return _ptr >= other._ptr; }
    difference_type operator+(const StaticArrayIterator &other) { return _ptr + other._ptr; }
    difference_type operator-(const StaticArrayIterator &other) { return _ptr - other._ptr; }

    friend bool operator==(const StaticArrayIterator &self, const StaticArrayIterator &sentinel) {
      return self._ptr == sentinel._ptr;
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

  struct no_init {};

  StaticArray(const std::size_t start, const std::size_t size, StaticArray &data)
      : StaticArray(size, data._data + start) {
    ASSERT(start + size <= data.size());
  }

  StaticArray(const std::size_t size, value_type *data) : _size{size}, _data{data} {}

  StaticArray(const std::size_t size, const value_type init_value = value_type()) { resize(size, init_value); }

  StaticArray(const std::size_t size, no_init) { resize_without_init(size); }

  StaticArray() {}

  StaticArray(const StaticArray &) = delete;
  StaticArray &operator=(const StaticArray &) = delete;
  StaticArray(StaticArray &&) noexcept = default;
  StaticArray &operator=(StaticArray &&) noexcept = default;

  //
  // Data access members
  //

  reference operator[](const size_type pos) {
    ASSERT(pos < _size);
    return _data[pos];
  }

  const_reference operator[](const size_type pos) const { return _data[pos]; }

  reference back() {
    ASSERT(_data);
    ASSERT(_size > 0);
    return _data[_size - 1];
  }

  const_reference back() const {
    ASSERT(_data);
    ASSERT(_size > 0);
    return _data[_size - 1];
  }

  reference front() {
    ASSERT(_data);
    ASSERT(_size > 0);
    return _data[0];
  }

  const_reference front() const {
    ASSERT(_data);
    ASSERT(_size > 0);
    return _data[0];
  }

  value_type *data() {
    ASSERT(_data);
    return _data;
  }

  const value_type *data() const {
    ASSERT(_data);
    return _data;
  }

  //
  // Iterators
  //

  iterator begin() {
    ASSERT(_data);
    return iterator(_data);
  }

  const_iterator cbegin() const {
    ASSERT(_data);
    return const_iterator(_data);
  }

  iterator end() { return iterator{_data + _size}; }

  const_iterator cend() const { return const_iterator{_data + _size}; }

  const_iterator begin() const { return cbegin(); }
  iterator end() const { return cend(); }

  void restrict(const std::size_t new_size) {
    ASSERT(new_size <= _size) << V(new_size) << V(_size);
    _unrestricted_size = _size;
    _size = new_size;
  }

  void unrestrict() { _size = _unrestricted_size; }

  //
  // Capacity
  //

  [[nodiscard]] bool empty() const { return _size == 0; }
  [[nodiscard]] size_type size() const { return _size; }

  void resize_without_init(const size_type size) {
    ASSERT(!_data);
    allocate_data(size);
  }

  void resize(const std::size_t size, no_init) { resize_without_init(size); }

  void resize(const size_type size, const value_type init_value = value_type(), const bool assign_parallel = true) {
    ASSERT(_data == _owned_data.get());
    resize_without_init(size);
    assign(size, init_value, assign_parallel);
  }

  void assign(const size_type count, const value_type value, const bool assign_parallel = true) {
    ASSERT(_data);

    if (assign_parallel) {
      const std::size_t step{std::max(count / std::thread::hardware_concurrency(), 1UL)};
      tbb::parallel_for(0UL, count, step, [&](const size_type i) {
        for (size_type j = i; j < std::min(i + step, count); ++j) { _data[j] = value; }
      });
    } else {
      for (std::size_t i = 0; i < count; ++i) { _data[i] = value; }
    }
  }

  parallel::tbb_unique_ptr<value_type> free() {
    _size = 0;
    _unrestricted_size = 0;
    _data = nullptr;
    return std::move(_owned_data);
  }

  void dump_frame() {
    LOG << "this=" << this << " _size=" << _size << " _unrestricted_size=" << _unrestricted_size << " _data=" << _data
        << " _owned_data=" << _owned_data.get();
  }

private:
  void allocate_data(const std::size_t size) {
    _owned_data = parallel::make_unique<value_type>(size);
    _data = _owned_data.get();
    _size = size;
    _unrestricted_size = _size;
  }

  size_type _size{0};
  size_type _unrestricted_size{0};
  parallel::tbb_unique_ptr<value_type> _owned_data{nullptr};
  value_type *_data{nullptr};
};

#if defined(TOOL) || defined(TEST)
template<typename T>
StaticArray<T> from_vec(const std::vector<T> &vec) {
  StaticArray<T> arr(vec.size());
  std::copy(vec.begin(), vec.end(), arr.begin());
  return arr;
}

template<typename T>
std::vector<T> to_vec(const StaticArray<T> &arr) {
  std::vector<T> vec(arr.size());
  std::copy(arr.begin(), arr.end(), vec.begin());
  return vec;
}
#endif // TOOL
} // namespace kaminpar