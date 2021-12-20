#pragma once

#include <memory>

namespace kaminpar {
// https://hackingcpp.com/cpp/recipe/uninitialized_numeric_array.html
template <typename T, typename Alloc = std::allocator<T>> class noinit_allocator : public Alloc {
public:
  template <typename U> struct rebind {
    using other = noinit_allocator<U, typename std::allocator_traits<Alloc>::template rebind_alloc<U>>;
  };

  using Alloc::Alloc;

  template <typename U> void construct(U *ptr) noexcept(std::is_nothrow_default_constructible_v<U>) {
    ::new (static_cast<void *>(ptr)) U;
  }

  template <typename U, typename... Args> void construct(U *ptr, Args &&...args) {
    std::allocator_traits<Alloc>::construct(static_cast<Alloc &>(*this), ptr, std::forward<Args>(args)...);
  }
};
} // namespace kaminpar