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
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/parallel/loops.h"

namespace kaminpar::parallel {

template <typename OutputIt, typename AlignedValueLambda>
std::size_t aligned_prefix_sum_seq(OutputIt begin, OutputIt end, AlignedValueLambda &&l) {
  std::size_t n = std::distance(begin, end);
  if (n == 0) {
    return 0;
  }

  *begin = 0;
  --n;
  if (n == 0) {
    return 0;
  }

  for (std::size_t i = 0; i < n; ++i) {
    const auto [alignment, value] = l(i);

    if (alignment > 0) {
      *(begin + i) += (alignment - (*(begin + i) % alignment)) % alignment;
      KASSERT(*(begin + i) % alignment == 0);
    }

    *(begin + i + 1) = *(begin + i) + value;
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
      0,
      n,
      [&](const std::size_t from, const std::size_t to, const int cpu) {
        for (std::size_t i = from; i < to; ++i) {
          const auto [alignment, value] = l(i);

          if (alignment > 0) {
            *(begin + i) += compute_alignment_offset(alignment, *(begin + i));
            KASSERT(*(begin + i) % alignment == 0);
          }

          if (i + 1 < to) {
            *(begin + i + 1) = *(begin + i) + value;
          }
        }
      }
  );

  std::vector<std::size_t> prefix_sums(ncpus);

  for (int cpu = 1; cpu < ncpus; ++cpu) {
    const auto [from, to] = math::compute_local_range<std::size_t>(n, ncpus, cpu);
    if (from == to) {
      continue;
    }

    const auto [alignment, value] = l(to - 1);
    const std::size_t last_offset = (*(begin + to - 1) += value);

    prefix_sums[cpu] = prefix_sums[cpu - 1] + last_offset +
                       compute_alignment_offset(alignment, prefix_sums[cpu - 1] + last_offset);
  }

  parallel::deterministic_for<std::size_t>(
      0,
      n,
      [&](const std::size_t from, const std::size_t to, const int cpu) {
        for (std::size_t i = from; i < to; ++i) {
          *(begin + i) += prefix_sums[cpu];
        }
      }
  );

  *(begin + n) += l(n - 1).second;

  return *(begin + n);
}

} // namespace kaminpar::parallel
