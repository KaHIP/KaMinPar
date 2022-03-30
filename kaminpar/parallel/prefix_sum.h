/*******************************************************************************
 * @file:   prefix_sum.h
 *
 * @author: Daniel Seemaier
 * @date:   30.03.2022
 * @brief:  Easy-to-use parallel prefix sum implementation.
 ******************************************************************************/
#pragma once

#include <tbb/blocked_range.h>
#include <tbb/parallel_scan.h>

namespace kaminpar::parallel {
template <typename InputIterator, typename OutputIterator>
void prefix_sum(InputIterator first, InputIterator last, OutputIterator result) {
  using size_t = std::size_t;                   // typename InputIterator::difference_type;
  using Value = std::decay_t<decltype(*first)>; // typename InputIterator::value_type;

  const size_t n = std::distance(first, last);
  tbb::parallel_scan(
      tbb::blocked_range<size_t>(0, n), Value(),
      [first, result](const tbb::blocked_range<size_t> &r, Value sum, bool is_final_scan) {
        Value temp = sum;
        for (auto i = r.begin(); i < r.end(); ++i) {
          temp += *(first + i);
          if (is_final_scan) {
            *(result + i) = temp;
          }
        }
        return temp;
      },
      [](Value left, Value right) { return left + right; });
}
} // namespace kaminpar
