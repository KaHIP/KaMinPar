/*******************************************************************************
 * Partial instantiation of `std::vector` using preallocated memory.
 *
 * @file:   preallocated_vector.h
 * @author: Daniel Seemaier
 * @date:   31.05.2022
 ******************************************************************************/
#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "kaminpar-common/assert.h"

namespace kaminpar {
template <typename T> class PreallocatedAllocator {
public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer = T *;

  PreallocatedAllocator(T *storage, const size_type size) throw()
      : _storage(storage),
        _size(size) {}

  PreallocatedAllocator(const PreallocatedAllocator &rhs) throw()
      : _storage(rhs._storage),
        _size(rhs._size) {}

  template <typename U> bool operator==(const PreallocatedAllocator<U> &other) noexcept {
    return _storage == other._storage && _size == other._size;
  }

  pointer allocate(const size_type n) {
    KASSERT(
        n == _size, "allocation request does not match the preallocated storage", assert::light
    );
    _size = 0;
    return _storage;
  }

  void deallocate(pointer, size_type) {}

  template <typename... Args> void construct(pointer, Args &&...) {}

  void destroy(pointer) {}

private:
  T *_storage;
  size_type _size;
};

template <typename T> using PreallocatedVector = std::vector<T, PreallocatedAllocator<T>>;

template <typename T>
PreallocatedVector<T>
make_preallocated_vector(T *storage, const std::size_t start, const std::size_t size) {
  return PreallocatedVector<T>(size, PreallocatedAllocator<T>(storage + start, size));
}

template <typename T>
PreallocatedVector<T> make_preallocated_vector(T *storage, const std::size_t size) {
  return make_preallocated_vector(storage, 0u, size);
}

template <typename Container>
auto make_preallocated_vector(Container &storage, const std::size_t start, const std::size_t size)
    -> PreallocatedVector<typename Container::value_type> {
  return make_preallocated_vector(std::data(storage), start, size);
}

template <typename Container>
auto make_preallocated_vector(Container &storage, const std::size_t size)
    -> PreallocatedVector<typename Container::value_type> {
  return make_preallocated_vector(storage, 0u, size);
}

template <typename Container>
auto make_preallocated_vector(Container &storage)
    -> PreallocatedVector<typename Container::value_type> {
  return make_preallocated_vector(storage, std::size(storage));
}
} // namespace kaminpar
