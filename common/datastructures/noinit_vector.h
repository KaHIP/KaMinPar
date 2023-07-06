/*******************************************************************************
 * Partial instantiation of `std::vector` that avoids value initialization.
 *
 * @file:   noinit_vector.h
 * @author: Daniel Seemaier
 * @date:   31.05.2022
 ******************************************************************************/
#pragma once

#include <memory>
#include <vector>

namespace kaminpar {
// https://hackingcpp.com/cpp/recipe/uninitialized_numeric_array.html
template <typename T, typename Allocator = std::allocator<T>>
class NoinitAllocator : public Allocator {
public:
  template <typename U> struct rebind {
    using other =
        NoinitAllocator<U, typename std::allocator_traits<Allocator>::template rebind_alloc<U>>;
  };

  using Allocator::Allocator;

  template <typename U>
  void construct(U *ptr) noexcept(std::is_nothrow_default_constructible_v<U>) {
    ::new (static_cast<void *>(ptr)) U;
  }

  template <typename U, typename... Args> void construct(U *ptr, Args &&...args) {
    std::allocator_traits<Allocator>::construct(
        static_cast<Allocator &>(*this), ptr, std::forward<Args>(args)...
    );
  }
};

template <typename T> using NoinitVector = std::vector<T, NoinitAllocator<T>>;
} // namespace kaminpar
