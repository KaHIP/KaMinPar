#pragma once

#include <functional>

#include <oneapi/tbb/concurrent_vector.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_sort.h>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/inline.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/parallel/loops.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::sparsification::utils {

StaticArray<EdgeID> sort_by_traget(const CSRGraph &g);

void for_edges_with_endpoints(
    const CSRGraph &g, std::function<void(EdgeID, NodeID, NodeID)> function
);
void for_upward_edges(const CSRGraph &g, std::function<void(EdgeID)>);
void for_downward_edges(const CSRGraph &g, std::function<void(EdgeID)>);

template <typename Lambda>
inline void parallel_for_edges_with_endpoints(const CSRGraph &g, Lambda function) {
  g.pfor_nodes([&](NodeID u) {
    g.neighbors(u, [&](EdgeID e, NodeID v, EdgeWeight) { function(e, u, v); });
  });
}
template <typename Lambda>
inline void parallel_for_upward_edges(const CSRGraph &g, Lambda function) {
  parallel_for_edges_with_endpoints(g, [&](EdgeID e, NodeID u, NodeID v) {
    if (u < v)
      function(e);
  });
}
template <typename Lambda>
inline void parallel_for_downward_edges(const CSRGraph &g, Lambda function) {
  p_for_edges_with_endpoints(g, [&](EdgeID e, NodeID u, NodeID v) {
    if (u > v)
      function(e);
  });
}

template <typename T, typename Iterator>
T sortselect_k_smallest(size_t k, Iterator begin, Iterator end) {
  size_t size = std::distance(begin, end);
  std::vector<T> sorted(size);
  for (size_t i = 0; i < size; i++) {
    sorted[i] = begin[i];
  }
  std::sort(sorted.begin(), sorted.end());
  return sorted[k - 1];
}

template <typename T> struct K_SmallestInfo {
  T value;
  std::size_t number_of_elements_smaller;
  std::size_t number_of_elements_equal;
};

template <typename T, typename Iterator>
K_SmallestInfo<T> quickselect_k_smallest(
    std::size_t k,
    Iterator begin,
    Iterator end,
    std::size_t number_of_smaller_elements_outside_partition = 0
);

template <typename T, typename Iterator> T median(Iterator begin, Iterator end) {
  size_t size = std::distance(begin, end);
  std::vector<T> sorted(size);
  for (std::size_t i = 0; i != size; i++) {
    sorted[i] = begin[i];
  }
  std::sort(sorted.begin(), sorted.end());
  if (size % 2 == 1) { // odd size
    return sorted[size / 2];
  } else {
    return (sorted[size / 2 - 1] + sorted[size / 2]) / 2;
  }
}

template <typename T, typename Iterator> T median_of_medians(Iterator begin, Iterator end) {
  const std::size_t size = std::distance(begin, end);
  if (size <= 10) {
    return median<T, Iterator>(begin, end);
  }

  const std::size_t number_of_sections = (size + 4) / 5;
  StaticArray<T> medians(number_of_sections);
  tbb::parallel_for(0ul, number_of_sections, [&](auto i) {
    medians[i] = median<T, Iterator>(begin + 5 * i, begin + std::min(5 * (i + 1), size));
  });

  return quickselect_k_smallest<T>((number_of_sections + 1) / 2, medians.begin(), medians.end())
      .value;
}

constexpr static std::size_t QUICKSELECT_BASE_CASE_SIZE = 20;

template <typename T, typename Iterator>
KAMINPAR_INLINE K_SmallestInfo<T> quickselect_k_smallest_base(
    const std::size_t k,
    Iterator begin,
    Iterator end,
    const std::size_t number_of_elements_outside_partition
) {
  const T k_smallest = sortselect_k_smallest<T>(k, begin, end);

  std::size_t number_equal = 0;
  std::size_t number_less = 0;
  for (auto x = begin; x != end; x++) {
    if (*x == k_smallest)
      number_equal++;
    else if (*x < k_smallest) {
      number_less++;
    }
  }

  return {k_smallest, number_less + number_of_elements_outside_partition, number_equal};
}

template <typename T, typename Iterator>
K_SmallestInfo<T> quickselect_k_smallest_iter(
    std::size_t k, Iterator begin, Iterator end, std::size_t number_of_elements_outside_partition
) {
  SCOPED_HEAP_PROFILER("Quickselect");

  constexpr static bool kUseBuffers = true;
  constexpr static bool kUseTrivialPivot = true;

  const std::size_t initial_size = std::distance(begin, end);

  bool aux_zeroed = true;
  RECORD("remap") StaticArray<EdgeID> remap;

  if constexpr (!kUseBuffers) {
    remap.resize(initial_size);
  }

  tbb::enumerable_thread_specific<std::size_t> thread_specific_number_equal;
  tbb::enumerable_thread_specific<std::size_t> thread_specific_number_less;
  tbb::enumerable_thread_specific<std::vector<T>> thread_specific_buffers;

  RECORD("current_elements") StaticArray<T> current_elements;
  RECORD("next_elements") StaticArray<T> next_elements;

  for (std::size_t size = initial_size; size > QUICKSELECT_BASE_CASE_SIZE;
       size = std::distance(begin, end)) {
    if (aux_zeroed) [[unlikely]] {
      aux_zeroed = false;
    } else {
      if constexpr (!kUseBuffers) {
        tbb::parallel_for<std::size_t>(0, size, [&](const std::size_t i) { remap[i] = 0; });
      }

      thread_specific_number_equal.clear();
      thread_specific_number_less.clear();
    }

    const T pivot = kUseTrivialPivot ? *begin : median_of_medians<T>(begin, end);

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

    if constexpr (kUseBuffers) {
      for (auto &buffer : thread_specific_buffers) {
        buffer.clear();
      }
    }

    if (k <= number_less) {
      if constexpr (kUseBuffers) {
        parallel::deterministic_for<std::size_t>(
            0,
            size,
            [&](const std::size_t from, const std::size_t to, int) {
              auto &buffer = thread_specific_buffers.local();

              for (std::size_t i = from; i < to; ++i) {
                if (begin[i] < pivot) {
                  buffer.push_back(begin[i]);
                }
              }
            }
        );
      } else {
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size), [&](const auto &r) {
          for (std::size_t i = r.begin(); i != r.end(); ++i) {
            if (begin[i] < pivot) {
              remap[i] = 1;
            }
          }
        });

        parallel::prefix_sum(remap.begin(), remap.begin() + size, remap.begin());
        KASSERT(remap[size - 1] == number_less, "prefix sum does not work", assert::always);
      }

      if (next_elements.size() < number_less) {
        next_elements.resize(number_less, static_array::noinit);
      }

      if constexpr (kUseBuffers) {
        std::size_t start = 0;
        for (const auto &buffer : thread_specific_buffers) {
          tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
            next_elements[start + i] = buffer[i];
          });
          start += buffer.size();
        }
      } else {
        tbb::parallel_for<std::size_t>(0, size, [&](const std::size_t i) {
          if (begin[i] < pivot) {
            next_elements[remap[i] - 1] = begin[i];
          }
        });
      }

      std::swap(next_elements, current_elements);

      begin = current_elements.begin();
      end = current_elements.begin() + number_less;
    } else if (k > number_less + number_equal) {
      if constexpr (kUseBuffers) {
        parallel::deterministic_for<std::size_t>(
            0,
            size,
            [&](const std::size_t from, const std::size_t to, int) {
              auto &buffer = thread_specific_buffers.local();

              for (std::size_t i = from; i < to; ++i) {
                if (begin[i] > pivot) {
                  buffer.push_back(begin[i]);
                }
              }
            }
        );
      } else {
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size), [&](const auto &r) {
          for (std::size_t i = r.begin(); i != r.end(); ++i) {
            if (begin[i] > pivot) {
              remap[i] = 1;
            }
          }
        });

        parallel::prefix_sum(remap.begin(), remap.begin() + size, remap.begin());
        KASSERT(
            remap[size - 1] == size - number_equal - number_less,
            "prefix sum does not work",
            assert::always
        );
      }

      const std::size_t number_greater = size - number_less - number_equal;
      if (next_elements.size() < number_greater) {
        next_elements.resize(number_greater, static_array::noinit);
      }

      if constexpr (kUseBuffers) {
        std::size_t start = 0;
        for (const auto &buffer : thread_specific_buffers) {
          tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
            next_elements[start + i] = buffer[i];
          });
          start += buffer.size();
        }
      } else {
        tbb::parallel_for<std::size_t>(0, size, [&](const std::size_t i) {
          if (begin[i] > pivot) {
            next_elements[remap[i] - 1] = begin[i];
          }
        });
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
K_SmallestInfo<T> quickselect_k_smallest(
    const std::size_t k,
    Iterator begin,
    Iterator end,
    const std::size_t number_of_elements_outside_partition
) {
  const std::size_t size = std::distance(begin, end);
  if (size <= QUICKSELECT_BASE_CASE_SIZE) {
    return quickselect_k_smallest_base<T>(k, begin, end, number_of_elements_outside_partition);
  }

  return quickselect_k_smallest_iter<T>(k, begin, end, number_of_elements_outside_partition);
}

} // namespace kaminpar::shm::sparsification::utils
