/*******************************************************************************
 * Utility functions for common computations.
 *
 * @file:   math.h
 * @author: Daniel Seemaier
 * @date:   17.06.2022
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "kaminpar-common/assert.h"

namespace kaminpar::math {
template <typename Int> bool is_square(const Int value) {
  const Int sqrt = std::sqrt(value);
  return sqrt * sqrt == value;
}

//! Checks whether `arg` is a power of 2.
template <typename T> constexpr bool is_power_of_2(const T arg) {
  return arg && ((arg & (arg - 1)) == 0);
}

//! With `UInt = uint32_t`, same as `static_cast<uint32_t>(std::log2(arg))`
template <typename T> T floor_log2(const T arg) {
  constexpr std::size_t arg_width{std::numeric_limits<T>::digits};

  auto log2{static_cast<T>(arg_width)};
  if constexpr (arg_width == std::numeric_limits<unsigned int>::digits) {
    log2 -= __builtin_clz(arg);
  } else {
    static_assert(
        arg_width == std::numeric_limits<unsigned long>::digits, "unsupported data type width"
    );
    log2 -= __builtin_clzl(arg);
  }

  return log2 - 1;
}

template <typename T> T floor2(const T arg) {
  return 1 << floor_log2(arg);
}

//! With `UInt = uint32_t`, same as
//! `static_cast<uint32_t>(std::ceil(std::log2(arg)))`
template <typename T> T ceil_log2(const T arg) {
  return floor_log2<T>(arg) + 1 - ((arg & (arg - 1)) == 0);
}

template <typename T> T ceil2(const T arg) {
  return 1 << ceil_log2(arg);
}

template <typename E>
double percentile(const std::vector<E> &sorted_sequence, const double percentile) {
  KASSERT([&] {
    for (std::size_t i = 1; i < sorted_sequence.size(); ++i) {
      if (sorted_sequence[i - 1] > sorted_sequence[i]) {
        return false;
      }
    }
    return true;
  }());
  KASSERT(0 <= percentile && percentile <= 1);

  return sorted_sequence[std::ceil(percentile * sorted_sequence.size()) - 1];
}

template <typename T> auto split_integral(const T value, const double ratio = 0.5) {
  return std::pair{
      static_cast<T>(std::ceil(value * ratio)), static_cast<T>(std::floor(value * (1.0 - ratio)))};
}

/**
 * Computes the first (inclusive) and last (exclusive) element that should be
 * processed on a PE.
 *
 * @param n Number of elements.
 * @param size Number of PEs that process the elements.
 * @param rank Rank of this PE.
 * @return First (inclusive) and last (exclusive) element that should be
 * processed by PE `rank`.
 */
template <typename Int>
std::pair<Int, Int> compute_local_range(const Int n, const Int size, const Int rank) {
  const Int chunk = n / size;
  const Int remainder = n % size;
  const Int from = rank * chunk + std::min<Int>(rank, remainder);
  const Int to = std::min<Int>(from + ((rank < remainder) ? chunk + 1 : chunk), n);
  return {from, to};
}

/**
 * Computes \c rank such that \c element is contained in the local range
 * computed by \c compute_local_range.
 *
 * @param n Number of elements.
 * @param size Number of PEs that process elements.
 * @param element A specific element.
 * @return The rank of the local range containing the element, i.e., the \c rank
 * parameter of \c compute_local_range such that its first return value is
 * less-or-equal to \c element and its second return value is larger than \c
 * element.
 */
template <typename Int>
std::size_t compute_local_range_rank(const Int n, const Int size, const Int element) {
  if (n <= size) {
    return element;
  } // special case if n is very small

  const Int c = n / size;
  const Int rem = n % size;
  const Int r0 = (element - rem) / c;
  return (element < rem || r0 < rem) ? element / (c + 1) : r0;
}

template <typename Int, typename Distribution>
std::size_t find_in_distribution(const Int value, const Distribution &distribution) {
  auto it = std::upper_bound(distribution.begin() + 1, distribution.end(), value);
  return std::distance(distribution.begin(), it) - 1;
}

/**
 * Given a total of n elements [0..n-1] across s PEs, compute a permutation such
 * that elements are ordered 0, s, 2s, ..., ss, 1, s+1, 2s+1, ..., ss+1, ...
 *
 * @tparam Int
 * @param n
 * @param size
 * @param element
 * @return
 */
template <typename Int> Int distribute_round_robin(const Int n, const Int size, const Int element) {
  const auto local =
      element - compute_local_range<Int>(n, size, compute_local_range_rank(n, size, element)).first;
  const auto owner = compute_local_range_rank(n, size, element);
  const auto ans = compute_local_range(n, size, local % size).first + (local / size) * size + owner;
  return ans;
}

template <typename Int>
std::pair<Int, Int> decode_grid_position(const Int pos, const Int num_columns) {
  const Int i = pos / num_columns;
  const Int j = pos % num_columns;
  return {i, j};
}

template <typename Int>
Int encode_grid_position(const Int row, const Int column, const Int num_columns) {
  return row * num_columns + column;
}

template <typename Container>
auto find_min(const Container &container) -> typename Container::value_type {
  return *std::min_element(container.begin(), container.end());
}

template <typename Container>
auto find_max(const Container &container) -> typename Container::value_type {
  return *std::max_element(container.begin(), container.end());
}

template <typename Container> double find_mean(const Container &container) {
  double sum = 0;
  for (const auto &e : container) {
    sum += e;
  }
  return sum / container.size();
}

template <typename Container>
auto find_min_mean_max(const Container &container)
    -> std::tuple<typename Container::value_type, double, typename Container::value_type> {
  return std::make_tuple(find_min(container), find_mean(container), find_max(container));
}

template <typename Int> Int create_mask(const int num_bits) {
  return (1 << num_bits) - 1;
}
} // namespace kaminpar::math
