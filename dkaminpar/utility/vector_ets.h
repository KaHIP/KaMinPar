/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
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