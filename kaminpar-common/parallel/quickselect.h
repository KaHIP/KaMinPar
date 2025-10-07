/*******************************************************************************
 * Parallel quickselect implementation.
 *
 * @file:   quickselect.h
 * @author: Dominik Rosch
 * @date:   24.08.2025
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <iterator>
#include <vector>

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/inline.h"
#include "kaminpar-common/parallel/loops.h"

namespace kaminpar {

template <typename T> struct QuickselectResult {
  T value;
  std::size_t number_of_elements_smaller;
  std::size_t number_of_elements_equal;
};

template <typename T, typename Iterator>
T sortselect_k_smallest(const std::size_t k, Iterator begin, Iterator end) {
  std::size_t size = std::distance(begin, end);
  std::vector<T> sorted(size);
  for (size_t i = 0; i < size; i++) {
    sorted[i] = begin[i];
  }
  std::sort(sorted.begin(), sorted.end());
  return sorted[k - 1];
}

constexpr static std::size_t QUICKSELECT_BASE_CASE_SIZE = 20;

template <typename T, typename Iterator>
KAMINPAR_INLINE QuickselectResult<T> quickselect_k_smallest_base(
    const std::size_t k,
    Iterator begin,
    Iterator end,
    const std::size_t number_of_elements_outside_partition
) {
  const T k_smallest = sortselect_k_smallest<T>(k, begin, end);

  std::size_t number_equal = 0;
  std::size_t number_less = 0;
  for (auto x = begin; x != end; x++) {
    if (*x == k_smallest) {
      number_equal++;
    } else if (*x < k_smallest) {
      number_less++;
    }
  }

  return {k_smallest, number_less + number_of_elements_outside_partition, number_equal};
}

template <typename T, typename Iterator>
QuickselectResult<T> quickselect_k_smallest_iter(
    std::size_t k, Iterator begin, Iterator end, std::size_t number_of_elements_outside_partition
) {
  SCOPED_HEAP_PROFILER("Quickselect");

  const std::size_t initial_size = std::distance(begin, end);
  bool aux_zeroed = true;

  tbb::enumerable_thread_specific<std::size_t> thread_specific_number_equal;
  tbb::enumerable_thread_specific<std::size_t> thread_specific_number_less;
  tbb::enumerable_thread_specific<std::vector<T>> thread_specific_buffers;

  StaticArray<T> current_elements;
  StaticArray<T> next_elements;

  for (std::size_t size = initial_size; size > QUICKSELECT_BASE_CASE_SIZE;
       size = std::distance(begin, end)) {
    if (aux_zeroed) [[unlikely]] {
      aux_zeroed = false;
    } else {
      thread_specific_number_equal.clear();
      thread_specific_number_less.clear();
    }

    const T pivot = *begin;

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size), [&](const auto &r) {
      std::size_t &less_counter = thread_specific_number_less.local();
      std::size_t &equal_counter = thread_specific_number_equal.local();

      for (std::size_t i = r.begin(); i != r.end(); ++i) {
        if (begin[i] < pivot) {
          ++less_counter;
        } else if (begin[i] == pivot) {
          ++equal_counter;
        }
      }
    });

    const std::size_t number_equal = thread_specific_number_equal.combine(std::plus{});
    const std::size_t number_less = thread_specific_number_less.combine(std::plus{});

    for (auto &buffer : thread_specific_buffers) {
      buffer.clear();
    }

    if (k <= number_less) {
      parallel::deterministic_for<std::size_t>(
          0, size, [&](const std::size_t from, const std::size_t to, int) {
            auto &buffer = thread_specific_buffers.local();

            for (std::size_t i = from; i < to; ++i) {
              if (begin[i] < pivot) {
                buffer.push_back(begin[i]);
              }
            }
          }
      );

      if (next_elements.size() < number_less) {
        next_elements.resize(number_less, static_array::noinit);
      }

      std::size_t start = 0;
      for (const auto &buffer : thread_specific_buffers) {
        tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
          next_elements[start + i] = buffer[i];
        });
        start += buffer.size();
      }

      std::swap(next_elements, current_elements);

      begin = current_elements.begin();
      end = current_elements.begin() + number_less;
    } else if (k > number_less + number_equal) {
      parallel::deterministic_for<std::size_t>(
          0, size, [&](const std::size_t from, const std::size_t to, int) {
            auto &buffer = thread_specific_buffers.local();

            for (std::size_t i = from; i < to; ++i) {
              if (begin[i] > pivot) {
                buffer.push_back(begin[i]);
              }
            }
          }
      );

      const std::size_t number_greater = size - number_less - number_equal;
      if (next_elements.size() < number_greater) {
        next_elements.resize(number_greater, static_array::noinit);
      }

      std::size_t start = 0;
      for (const auto &buffer : thread_specific_buffers) {
        tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
          next_elements[start + i] = buffer[i];
        });
        start += buffer.size();
      }

      std::swap(next_elements, current_elements);

      k -= number_equal + number_less;
      begin = current_elements.begin();
      end = current_elements.begin() + number_greater;
      number_of_elements_outside_partition += number_less + number_equal;
    } else {
      return {pivot, number_less + number_of_elements_outside_partition, number_equal};
    }
  }

  return quickselect_k_smallest_base<T>(k, begin, end, number_of_elements_outside_partition);
}

template <typename T, typename Iterator>
QuickselectResult<T> quickselect_k_smallest(
    const std::size_t k,
    Iterator begin,
    Iterator end,
    const std::size_t number_of_elements_outside_partition = 0
) {
  const std::size_t size = std::distance(begin, end);
  if (size <= QUICKSELECT_BASE_CASE_SIZE) {
    return quickselect_k_smallest_base<T>(k, begin, end, number_of_elements_outside_partition);
  }

  return quickselect_k_smallest_iter<T>(k, begin, end, number_of_elements_outside_partition);
}

} // namespace kaminpar
