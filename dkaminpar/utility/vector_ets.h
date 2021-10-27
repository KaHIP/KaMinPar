/*******************************************************************************
 * @file:   vector_ets.h
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Vector of thread-local data that can be combined.
 ******************************************************************************/
#pragma once

#include "dkaminpar/distributed_definitions.h"

#include <tbb/combinable.h>

namespace dkaminpar::parallel {
template<typename T, template<typename> typename Container = scalable_vector>
class vector_ets {
public:
  explicit vector_ets(const std::size_t size) : _size{size}, _ets{[size] { return Container<T>(size); }} {}

  vector_ets(const vector_ets &) = delete;
  vector_ets(vector_ets &&) noexcept = default;
  vector_ets &operator=(const vector_ets &) = delete;
  vector_ets &operator=(vector_ets &&) noexcept = delete;

  auto &local() { return _ets.local(); }

  template<typename BinaryOp>
  Container<T> combine(BinaryOp &&op) {
    return _ets.combine([&](const Container<T> &a, const Container<T> &b) {
      Container<T> ans(_size);
      for (std::size_t i = 0; i < _size; ++i) { ans[i] = op(a[i], b[i]); }
      return ans;
    });
  }

private:
  std::size_t _size;
  tbb::combinable<Container<T>> _ets;
};
} // namespace dkaminpar::parallel