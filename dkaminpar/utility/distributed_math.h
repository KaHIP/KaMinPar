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

#include <concepts>
#include <utility>

namespace dkaminpar::math {
/**
 * Computes the first (inclusive) and last (exclusive) element that should be processed on a PE.
 *
 * @param n Number of elements.
 * @param rank Rank of this PE.
 * @param size Number of PEs that process the elements.
 * @return First (inclusive) and last (exclusive) element that should be processed by PE `rank`.
 */
template<std::integral Int>
std::pair<Int, Int> compute_local_range(const Int n, const Int size, const Int rank) {
  const Int chunk = n / size;
  const Int remainder = n % size;
  const Int from = rank * chunk + std::min<Int>(rank, remainder);
  const Int to = std::min<Int>(from + ((rank < remainder) ? chunk + 1 : chunk), n);
  return {from, to};
}
} // namespace dkaminpar::math