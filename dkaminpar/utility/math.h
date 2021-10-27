/*******************************************************************************
 * @file:   math.h
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Math utility functions.
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