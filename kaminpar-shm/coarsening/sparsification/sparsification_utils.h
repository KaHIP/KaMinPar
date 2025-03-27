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
    // TODO: make parallel again
    g.neighbors(u, [&](EdgeID e, NodeID v, EdgeWeight e_weight) { function(e, u, v); });
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
  size_t number_of_elements_smaller;
  size_t number_of_elemtns_equal;
};
template <typename T, typename Iterator>
K_SmallestInfo<T> quickselect_k_smallest(size_t k, Iterator begin, Iterator end);

template <typename T, typename Iterator> T median(Iterator begin, Iterator end) {
  size_t size = std::distance(begin, end);
  std::vector<T> sorted(size);
  for (auto i = 0; i != size; i++) {
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
KAMINPAR_INLINE K_SmallestInfo<T>
quickselect_k_smallest_base(const std::size_t k, Iterator begin, Iterator end) {
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

  return {k_smallest, number_less, number_equal};
}

template <typename T, typename Iterator>
K_SmallestInfo<T> quickselect_k_smallest_iter(std::size_t k, Iterator begin, Iterator end) {
  SCOPED_HEAP_PROFILER("Quickselect");

  const std::size_t initial_size = std::distance(begin, end);

  bool aux_zeroed = true;
  RECORD("remap") StaticArray<EdgeID> remap(initial_size);

  tbb::enumerable_thread_specific<std::size_t> thread_specific_number_equal;
  tbb::enumerable_thread_specific<std::size_t> thread_specific_number_less;

  RECORD("current_elements") StaticArray<T> current_elements;
  RECORD("next_elements") StaticArray<T> next_elements;

  for (std::size_t size = initial_size; size > QUICKSELECT_BASE_CASE_SIZE;
       size = std::distance(begin, end)) {
    if (aux_zeroed) [[unlikely]] {
      aux_zeroed = false;
    } else {
      tbb::parallel_for<std::size_t>(0, size, [&](const std::size_t i) { remap[i] = 0; });

      thread_specific_number_equal.clear();
      thread_specific_number_less.clear();
    }

    const T pivot = *begin; // median_of_medians<T>(begin, end);

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

    if (k <= number_less) {
      tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size), [&](const auto &r) {
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
          if (begin[i] < pivot) {
            remap[i] = 1;
          }
        }
      });

      parallel::prefix_sum(remap.begin(), remap.begin() + size, remap.begin());
      KASSERT(remap[size - 1] == number_less, "prefix sum does not work", assert::always);

      if (next_elements.size() < number_less) {
        next_elements.resize(number_less, static_array::noinit);
      }

      tbb::parallel_for<std::size_t>(0, size, [&](const std::size_t i) {
        if (begin[i] < pivot) {
          next_elements[remap[i] - 1] = begin[i];
        }
      });

      std::swap(next_elements, current_elements);

      begin = current_elements.cbegin();
      end = current_elements.cbegin() + number_less;
    } else if (k > number_less + number_equal) {
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

      const std::size_t number_greater = size - number_less - number_equal;
      if (next_elements.size() < number_greater) {
        next_elements.resize(number_greater, static_array::noinit);
      }

      tbb::parallel_for<std::size_t>(0, size, [&](const std::size_t i) {
        if (begin[i] > pivot) {
          next_elements[remap[i] - 1] = begin[i];
        }
      });

      std::swap(next_elements, current_elements);

      k -= number_equal + number_less;
      begin = current_elements.cbegin();
      end = current_elements.cbegin() + number_greater;
    } else {
      return {pivot, number_less, number_equal};
    }
  }

  return quickselect_k_smallest_base<T>(k, begin, end);
}

template <typename T, typename Iterator>
K_SmallestInfo<T> quickselect_k_smallest_rec(const std::size_t k, Iterator begin, Iterator end) {
  const std::size_t size = std::distance(begin, end);
  if (size <= QUICKSELECT_BASE_CASE_SIZE) {
    return quickselect_k_smallest_base<T>(k, begin, end);
  }

  StaticArray<std::size_t> less(size);
  StaticArray<std::size_t> greater(size);

  const T pivot = median_of_medians<T>(begin, end);

  tbb::enumerable_thread_specific<std::size_t> thread_specific_number_equal;
  tbb::enumerable_thread_specific<std::size_t> thread_specific_number_less;

  tbb::parallel_for(tbb::blocked_range<std::size_t>(0, size), [&](const auto &r) {
    std::size_t &less_counter = thread_specific_number_less.local();
    std::size_t &equal_counter = thread_specific_number_equal.local();

    for (std::size_t i = r.begin(); i != r.end(); ++i) {
      if (begin[i] < pivot) {
        less[i] = 1;
        ++less_counter;
      } else if (begin[i] > pivot) {
        greater[i] = 1;
      } else {
        ++equal_counter;
      }
    }
  });

  const std::size_t number_equal = thread_specific_number_equal.combine(std::plus{});
  const std::size_t number_less = thread_specific_number_less.combine(std::plus{});

  if (k <= number_less) {
    parallel::prefix_sum(less.begin(), less.begin() + size, less.begin());
    KASSERT(less[size - 1] == number_less, "prefix sum does not work", assert::always);

    StaticArray<T> elements_less(number_less);
    tbb::parallel_for<std::size_t>(0, size, [&](const std::size_t i) {
      if (begin[i] < pivot) {
        elements_less[less[i] - 1] = begin[i];
      }
    });

    return quickselect_k_smallest_rec<T>(k, elements_less.begin(), elements_less.end());
  } else if (k > number_less + number_equal) {
    parallel::prefix_sum(greater.begin(), greater.begin() + size, greater.begin());
    KASSERT(
        greater[size - 1] == size - number_equal - number_less,
        "prefix sum does not work",
        assert::always
    );

    StaticArray<T> elements_greater(size - number_equal - number_less);
    tbb::parallel_for<std::size_t>(0, size, [&](const std::size_t i) {
      if (begin[i] > pivot) {
        elements_greater[greater[i] - 1] = begin[i];
      }
    });

    return quickselect_k_smallest_rec<T>(
        k - number_equal - number_less, elements_greater.begin(), elements_greater.end()
    );
  } else {
    return {pivot, number_less, number_equal};
  }
}

template <typename T, typename Iterator>
K_SmallestInfo<T> quickselect_k_smallest(const std::size_t k, Iterator begin, Iterator end) {
  const std::size_t size = std::distance(begin, end);
  if (size <= QUICKSELECT_BASE_CASE_SIZE) {
    return quickselect_k_smallest_base<T>(k, begin, end);
  }

  // return quickselect_k_smallest_rec<T>(k, begin, end);
  return quickselect_k_smallest_iter<T>(k, begin, end);
}

} // namespace kaminpar::shm::sparsification::utils
