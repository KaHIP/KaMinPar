/*******************************************************************************
 * Fixed-size array backed by its own memory or a provided memory region.
 *
 * @file:   static_array.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <thread>
#include <vector>

#include <tbb/parallel_for.h>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/constexpr_utils.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/parallel/tbb_malloc.h"

// Threshold for using transparent huge pages for large memory allocations (unit: number of
// elements, *not* number of bytes -- @todo change this?)
#define KAMINPAR_THP_THRESHOLD 1024 * 1024 * 64

namespace kaminpar {
namespace static_array {
//! Tag for allocating memory, but not touching it. Without this tag, memory is initialized to the
//! default value for the given type.
constexpr struct noinit_t {
} noinit;

//! Tag for small memory allocations that should never be backed by transparent huge pages.
constexpr struct small_t {
} small;

//! Tag for uninitialized and overcommited memory allocations that are manually tracked.
constexpr struct overcommit_t {
} overcommit;

//! Tag for initializing memory sequentially. Without this tag, memory will be initialized by a
//! parallel loop. Has no effect in the presence of the noinit tag.
constexpr struct seq_t {
} seq;
} // namespace static_array

template <typename T> class StaticArray {
public:
  template <bool is_const> class StaticArrayIterator {
  public:
    using iterator_category = std::contiguous_iterator_tag;
    using value_type = T;
    using reference = std::conditional_t<is_const, const T &, T &>;
    using pointer = std::conditional_t<is_const, const T *, T *>;
    using difference_type = std::ptrdiff_t;

    StaticArrayIterator() : _ptr(nullptr) {}
    StaticArrayIterator(pointer ptr) : _ptr(ptr) {}

    StaticArrayIterator(const StaticArrayIterator &other) = default;
    StaticArrayIterator &operator=(const StaticArrayIterator &other) = default;

    reference operator*() const {
      return *_ptr;
    }

    pointer operator->() const {
      return _ptr;
    }

    reference operator[](const difference_type &n) const {
      return _ptr[n];
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

    StaticArrayIterator &operator+=(const difference_type &n) {
      return _ptr += n, *this;
    }

    StaticArrayIterator &operator-=(const difference_type &n) {
      return _ptr -= n, *this;
    }

    friend bool operator==(const StaticArrayIterator &lhs, const StaticArrayIterator &rhs) {
      return lhs._ptr == rhs._ptr;
    }

    friend bool operator!=(const StaticArrayIterator &lhs, const StaticArrayIterator &rhs) {
      return lhs._ptr != rhs._ptr;
    }

    friend bool operator>(const StaticArrayIterator &lhs, const StaticArrayIterator &rhs) {
      return lhs._ptr > rhs._ptr;
    }

    friend bool operator<(const StaticArrayIterator &lhs, const StaticArrayIterator &rhs) {
      return lhs._ptr < rhs._ptr;
    }

    friend bool operator<=(const StaticArrayIterator &lhs, const StaticArrayIterator &rhs) {
      return lhs._ptr <= rhs._ptr;
    }

    friend bool operator>=(const StaticArrayIterator &lhs, const StaticArrayIterator &rhs) {
      return lhs._ptr >= rhs._ptr;
    }

    friend difference_type
    operator-(const StaticArrayIterator &lhs, const StaticArrayIterator &rhs) {
      return lhs._ptr - rhs._ptr;
    }

    friend StaticArrayIterator operator+(const StaticArrayIterator &it, const difference_type n) {
      return {it._ptr + n};
    }

    friend StaticArrayIterator operator-(const StaticArrayIterator &it, const difference_type n) {
      return {it._ptr - n};
    }

    friend StaticArrayIterator operator+(const difference_type n, const StaticArrayIterator &it) {
      return {it._ptr + n};
    }

  private:
    pointer _ptr;
  };

public:
  using value_type = T;
  using size_type = std::size_t;
  using reference = T &;
  using const_reference = const T &;
  using iterator = StaticArrayIterator<false>;
  using const_iterator = StaticArrayIterator<true>;

  StaticArray(const std::size_t size, value_type *data) : _size(size), _data(data) {
    RECORD_DATA_STRUCT(0, _struct);
  }

  StaticArray(const std::size_t size, heap_profiler::unique_ptr<T> storage)
      : _size(size),
        _overcommited_data(std::move(storage)),
        _data(_overcommited_data.get()) {
    RECORD_DATA_STRUCT(size * sizeof(T), _struct);
  }

  template <typename... Tags>
  StaticArray(const std::size_t size, const value_type init_value, Tags... tags) {
    RECORD_DATA_STRUCT(0, _struct);
    resize(size, init_value, std::forward<Tags>(tags)...);
  }

  template <typename... Tags> StaticArray(const std::size_t size, Tags... tags) {
    RECORD_DATA_STRUCT(0, _struct);
    resize(size, value_type(), std::forward<Tags>(tags)...);
  }

  template <typename Iterator>
  StaticArray(Iterator first, Iterator last)
      : StaticArray(std::distance(first, last), static_array::noinit) {
    tbb::parallel_for<std::size_t>(0, _size, [&](const std::size_t i) { _data[i] = *(first + i); });
  }

  StaticArray() {
    RECORD_DATA_STRUCT(0, _struct);
  }

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

  [[nodiscard]] bool is_span() const {
    return _owned_data.get() == nullptr && _overcommited_data.get() == nullptr;
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
    KASSERT(_data && _size > 0u);
    return _data[_size - 1];
  }

  const_reference back() const {
    KASSERT(_data && _size > 0u);
    return _data[_size - 1];
  }

  reference front() {
    KASSERT(_data && _size > 0u);
    return _data[0];
  }

  const_reference front() const {
    KASSERT(_data && _size > 0u);
    return _data[0];
  }

  value_type *data() {
    KASSERT(_data || _size == 0);
    return _data;
  }

  const value_type *data() const {
    KASSERT(_data || _size == 0);
    return _data;
  }

  //
  // Iterators
  //

  iterator begin() {
    KASSERT(_data || _size == 0);
    return iterator(_data);
  }

  const_iterator begin() const {
    KASSERT(_data || _size == 0);
    return const_iterator(_data);
  }

  const_iterator cbegin() const {
    return begin();
  }

  iterator end() {
    KASSERT(_data || _size == 0);
    return iterator(_data + _size);
  }

  const_iterator end() const {
    KASSERT(_data || _size == 0);
    return const_iterator(_data + _size);
  }

  const_iterator cend() const {
    return end();
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

  template <typename... Tags> void resize(const std::size_t size, Tags... tags) {
    resize(size, value_type(), std::forward<Tags>(tags)...);
  }

  template <typename... Tags>
  void resize(const std::size_t size, const value_type init_value, Tags...) {
    KASSERT(
        _data == _owned_data.get() || _data == _overcommited_data.get(),
        "cannot resize span",
        assert::always
    );

    const bool overcommit = kHeapProfiling && contains_tag_v<static_array::overcommit_t, Tags...>;
    const bool use_thp =
        size >= KAMINPAR_THP_THRESHOLD && !contains_tag_v<static_array::small_t, Tags...>;

    allocate_data(size, overcommit, use_thp);

    if constexpr (!contains_tag_v<static_array::noinit_t, Tags...>) {
      if constexpr (contains_tag_v<static_array::seq_t, Tags...>) {
        assign(size, init_value, false);
      } else {
        assign(size, init_value, true);
      }
    }
  }

  void assign(const size_type count, const value_type value, const bool assign_parallel = true) {
    KASSERT(_data || count == 0);

    if (assign_parallel) {
      const std::size_t step = std::max(count / std::thread::hardware_concurrency(), 1UL);
      tbb::parallel_for<std::size_t>(0, count, step, [&](const size_type i) noexcept {
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

  void free() {
    _size = 0;
    _unrestricted_size = 0;
    _data = nullptr;

    _owned_data.reset();
    _overcommited_data.reset();
  }

private:
  void allocate_data(const std::size_t size, const bool overcommit, const bool thp) {
    // Before allocating the new memory, free the old memory to prevent both from being held in
    // memory at the same time
    if (_owned_data != nullptr) {
      _owned_data.reset();
    }
    if (_overcommited_data != nullptr) {
      _overcommited_data.reset();
    }

    if (overcommit) {
      _overcommited_data = heap_profiler::overcommit_memory<value_type>(size);
      _data = _overcommited_data.get();
    } else {
      _owned_data = parallel::make_unique<value_type>(size, thp);
      _data = _owned_data.get();
    }

    _size = size;
    _unrestricted_size = _size;

    IF_HEAP_PROFILING(_struct->size = std::max(_struct->size, size * sizeof(value_type)));
  }

  size_type _size = 0;
  size_type _unrestricted_size = 0;
  parallel::tbb_unique_ptr<value_type> _owned_data = nullptr;
  heap_profiler::unique_ptr<value_type> _overcommited_data = nullptr;
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
