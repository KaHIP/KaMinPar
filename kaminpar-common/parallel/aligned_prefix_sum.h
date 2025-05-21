/*******************************************************************************
 * Computes the prefix sum over provided values while ensuring that each value
 * satisfies some alignment.
 *
 * @file:   aligned_prefix_sum.h
 * @author: Daniel Seemaier
 * @date:   26.08.2024
 ******************************************************************************/
#pragma once

#include <iterator>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/math.h"
#include "kaminpar-common/parallel/loops.h"

namespace kaminpar::parallel {

template <typename OutputIt, typename AlignedValueLambda>
std::size_t aligned_prefix_sum_seq(OutputIt begin, OutputIt end, AlignedValueLambda &&l) {
  std::size_t n = std::distance(begin, end);
  if (n == 0) {
    return 0;
  }

  --n;
  if (n == 0) {
    return 0;
  }

  for (std::size_t i = 0; i < n; ++i) {
    const auto [alignment, value] = l(i);

    if (i > 0 && alignment > 0) {
      *(begin + i) += (alignment - (*(begin + i) % alignment)) % alignment;
      KASSERT(static_cast<std::uint64_t>(*(begin + i) % alignment) == 0u);
    }

    *(begin + i + 1) = (i > 0 ? *(begin + i) : 0) + value;
  }

  const auto [last_alignment, last_value] = l(n);
  if (last_alignment > 0) {
    *(begin + n) += (last_alignment - (*(begin + n) % last_alignment)) % last_alignment;
  }

  return *(begin + n);
}

template <typename OutputIt, typename AlignedValueLambda>
std::size_t aligned_prefix_sum(OutputIt begin, OutputIt end, AlignedValueLambda &&l) {
  std::size_t n = std::distance(begin, end);
  if (n == 0) {
    return 0;
  }

  *begin = 0;
  --n;
  if (n == 0) {
    return 0;
  }

  auto compute_alignment_offset = [&](const std::size_t alignment, const std::size_t value) {
    return (alignment - (value % alignment)) % alignment;
  };

  const int ncpus = parallel::deterministic_for<std::size_t>(
      0, n, [&](const std::size_t from, const std::size_t to, int) {
        aligned_prefix_sum_seq(begin + from, begin + to + 1, [&](const std::size_t i) {
          return l(from + i);
        });
      }
  );

  std::vector<std::size_t> prefix_sums(ncpus);

  for (int cpu = 1; cpu < ncpus; ++cpu) {
    const auto [from, to] = math::compute_local_range<std::size_t>(n, ncpus, cpu);
    if (from == to) {
      continue;
    }

    const auto value = prefix_sums[cpu - 1] + *(begin + from);
    prefix_sums[cpu] = value + compute_alignment_offset(8, value);
  }

  parallel::deterministic_for<std::size_t>(
      0, n, [&](const std::size_t from, const std::size_t to, const int cpu) {
        for (std::size_t i = from + 1; i < to + 1; ++i) {
          *(begin + i) += prefix_sums[cpu];
        }
      }
  );

  return *(begin + n) + l(n).second;
}

} // namespace kaminpar::parallel
