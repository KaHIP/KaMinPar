/*******************************************************************************
 * @file:   math.h
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Math utility functions.
 ******************************************************************************/
#pragma once

#include "kaminpar/definitions.h"

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
template <std::integral Int> std::pair<Int, Int> compute_local_range(const Int n, const Int size, const Int rank) {
  const Int chunk = n / size;
  const Int remainder = n % size;
  const Int from = rank * chunk + std::min<Int>(rank, remainder);
  const Int to = std::min<Int>(from + ((rank < remainder) ? chunk + 1 : chunk), n);
  return {from, to};
}

/**
 * Computes \c rank such that \c element is contained in the local range computed by \c compute_local_range.
 *
 * @param n Number of elements.
 * @param size Number of PEs that process elements.
 * @param element A specific element.
 * @return The rank of the local range containing the element, i.e., the \c rank parameter of \c compute_local_range
 * such that its first return value is less-or-equal to \c element and its second return value is larger than
 * \c element.
 */
template <std::integral Int> std::size_t compute_local_range_rank(const Int n, const Int size, const Int element) {
  if (n <= size) {
    return element;
  } // special case if n is very small

  const Int c = n / size;
  const Int rem = n % size;
  const Int r0 = (element - rem) / c;
  return (element < rem || r0 < rem) ? element / (c + 1) : r0;
}

template <std::integral Int> std::size_t find_in_distribution(const Int value, const auto &distribution) {
  ASSERT(value < distribution.back()) << V(value) << V(distribution);
  auto it = std::upper_bound(distribution.begin() + 1, distribution.end(), value);
  return std::distance(distribution.begin(), it) - 1;
}

/**
 * Given a total of n elements [0..n-1] across s PEs, compute a permutation such that elements are ordered
 * 0, s, 2s, ..., ss, 1, s+1, 2s+1, ..., ss+1, ...
 *
 * @tparam Int
 * @param n
 * @param size
 * @param element
 * @return
 */
template <std::integral Int> Int distribute_round_robin(const Int n, const Int size, const Int element) {
  const auto divisor = n / size;
  const auto local = element - compute_local_range(n, size, compute_local_range_rank(n, size, element)).first;
  const auto owner = compute_local_range_rank(n, size, element);
  const auto ans = compute_local_range(n, size, local % size).first + (local / size) * size + owner;
  return ans;
}
} // namespace dkaminpar::math