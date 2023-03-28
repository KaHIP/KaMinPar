/*******************************************************************************
 * @file:   vector_ets.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Vector of combinable thread-local data.
 ******************************************************************************/
#pragma once

#include <vector>

#include <tbb/cache_aligned_allocator.h>
#include <tbb/combinable.h>

namespace kaminpar::parallel {
template <typename T> class vector_ets {
public:
  using Container = std::vector<T, tbb::cache_aligned_allocator<T>>;

  explicit vector_ets(const std::size_t size)
      : _size{size},
        _ets{[size] {
          return Container(size);
        }} {}

  vector_ets(const vector_ets &) = delete;
  vector_ets(vector_ets &&) noexcept = default;

  vector_ets &operator=(const vector_ets &) = delete;
  vector_ets &operator=(vector_ets &&) noexcept = delete;

  auto &local() {
    return _ets.local();
  }

  template <typename BinaryOp> Container combine(BinaryOp &&op) {
    return _ets.combine([&](const Container &a, const Container &b) {
      Container ans(_size);
      for (std::size_t i = 0; i < _size; ++i) {
        ans[i] = op(a[i], b[i]);
      }
      return ans;
    });
  }

private:
  std::size_t _size;
  tbb::combinable<Container> _ets;
};
} // namespace kaminpar::parallel
