/*******************************************************************************
 * Partial instantiation of `std::vector` that limits growing.
 *
 * @file:   maxsize_vector.h
 * @author: Daniel Salwasser
 * @date:   16.06.2024
 ******************************************************************************/
#pragma once

#include <memory>
#include <vector>

namespace kaminpar {

template <typename T> class MaxCapacityAllocator : public std::allocator<T> {
  using Base = std::allocator<T>;

public:
  using value_type = Base::value_type;
  using size_type = Base::size_type;
  using difference_type = Base::difference_type;

  MaxCapacityAllocator(const std::size_t max_capacity) noexcept
      : Base(),
        _max_capacity(max_capacity) {}

  template <typename U>
  MaxCapacityAllocator(const MaxCapacityAllocator<U> &allocator) noexcept : Base(allocator) {
    _max_capacity = allocator._max_capacity;
  }

  template <class U> bool operator==(const MaxCapacityAllocator<U> &other) const noexcept {
    return _max_capacity == other._max_capacity;
  }

  T *allocate(size_type n) {
    return Base::allocate(std::min(n, _max_capacity));
  }

  void deallocate(T *p, size_type n) {
    return Base::deallocate(p, n);
  }

private:
  std::size_t _max_capacity;
};

template <typename T> class MaxSizeVector : private std::vector<T, MaxCapacityAllocator<T>> {
  using Base = std::vector<T, MaxCapacityAllocator<T>>;

public:
  using value_type = Base::value_type;
  using allocator_type = Base::allocator_type;
  using size_type = Base::size_type;
  using difference_type = Base::difference_type;

  using reference = Base::reference;
  using const_reference = Base::const_reference;
  using pointer = Base::pointer;
  using const_pointer = Base::const_pointer;

  using iterator = Base::iterator;
  using const_iterator = Base::const_iterator;
  using reverse_iterator = Base::reverse_iterator;
  using const_reverse_iterator = Base::const_reverse_iterator;

  using Base::assign;
  using Base::get_allocator;

  using Base::at;
  using Base::operator[];
  using Base::back;
  using Base::data;
  using Base::front;

  using Base::begin;
  using Base::cbegin;
  using Base::cend;
  using Base::crbegin;
  using Base::crend;
  using Base::end;
  using Base::rbegin;
  using Base::rend;

  using Base::capacity;
  using Base::empty;
  using Base::max_size;
  using Base::reserve;
  using Base::shrink_to_fit;
  using Base::size;

  using Base::clear;
  using Base::emplace;
  using Base::emplace_back;
  using Base::erase;
  using Base::insert;
  using Base::pop_back;
  using Base::push_back;
  using Base::resize;
  using Base::swap;

  MaxSizeVector(const std::size_t max_capacity)
      : Base(MaxCapacityAllocator<T>(max_capacity)),
        _max_capacity(max_capacity) {}

  size_type max_size() const {
    return std::min(Base::max_size(), _max_capacity);
  }

  Base::size_type capacity() const {
    return std::min(Base::capacity(), _max_capacity);
  }

private:
  std::size_t _max_capacity;
};

} // namespace kaminpar
