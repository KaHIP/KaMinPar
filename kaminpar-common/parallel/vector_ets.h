/*******************************************************************************
 * Vector of combinable thread-local data.
 *
 * @file:   vector_ets.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#pragma once

#include <tbb/combinable.h>

#include "kaminpar-common/datastructures/cache_aligned_vector.h"

namespace kaminpar::parallel {

template <typename T> class vector_ets {
public:
  using Container = CacheAlignedVector<T>;

  explicit vector_ets(const std::size_t size)
      : _size(size),
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
