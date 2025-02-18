#pragma once

#include <functional>

#include <oneapi/tbb/concurrent_vector.h>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <oneapi/tbb/parallel_sort.h>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/random.h"

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
    g.neighbors(u,  [&](EdgeID e, NodeID v, EdgeWeight e_weight) {
      function(e, u, v);
    });
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


template <typename T>
struct K_SmallestInfo {
  T value;
  size_t number_of_elements_smaller;
  size_t number_of_elemtns_equal;
};
template <typename T, typename Iterator>
K_SmallestInfo<T> quickselect_k_smallest(size_t k, Iterator begin, Iterator end) ;

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
  size_t size = std::distance(begin, end);
  if (size <= 10)
    return median<T, Iterator>(begin, end);

  size_t number_of_sections = (size + 4) / 5;
  StaticArray<T> medians(number_of_sections);
  tbb::parallel_for(0ul, number_of_sections, [&](auto i) {
    medians[i] = median<T, Iterator>(begin + 5 * i, begin + std::min(5 * (i + 1), size));
  });

  return quickselect_k_smallest<T, typename StaticArray<T>::iterator>(
      (number_of_sections + 1) / 2, medians.begin(), medians.end()
  ).value;
}

template <typename T, typename Iterator>
K_SmallestInfo<T> quickselect_k_smallest(size_t k, Iterator begin, Iterator end) {
  size_t size = std::distance(begin, end);
  if (size <= 20) {
    T k_smallest = sortselect_k_smallest<T, Iterator>(k, begin, end);
    size_t number_equal = 0; size_t number_less;
    for (auto x = begin; x != end; x++) {
      if (*x == k_smallest)
        number_equal++;
      else if (*x < k_smallest) {
        number_less++;
      }
    }
    return {k_smallest, number_less, number_equal};
  }
  T pivot = median_of_medians<T,Iterator>(begin,end);

  StaticArray<size_t> less(size);
  StaticArray<size_t> greater(size);
  tbb::enumerable_thread_specific<size_t> thread_specific_number_equal;
  tbb::enumerable_thread_specific<size_t> thread_specific_number_less;
  tbb::parallel_for(0ul, size, [&](size_t i) {
    if (begin[i] < pivot) {
      less[i] = 1;
      thread_specific_number_less.local()++;
    } else if (begin[i] > pivot) {
      greater[i] = 1;
    } else {
      thread_specific_number_equal.local()++;
    }
  });

  auto add = [](size_t a, size_t b) {
    return a + b;
  };
  size_t number_equal = thread_specific_number_equal.combine(add);
  size_t number_less = thread_specific_number_less.combine(add);

  if (k <= number_less) {
    parallel::prefix_sum(less.begin(), less.end(), less.begin());
    KASSERT(less[size-1] == number_less, "prefix sum does not work", assert::always);

    StaticArray<T> elements_less(number_less);
    tbb::parallel_for(0ul, size, [&](auto i) {
      if (begin[i] < pivot) {
        elements_less[less[i] - 1] = begin[i];
      }
    });

    return quickselect_k_smallest<T>(k, elements_less.begin(), elements_less.end());
  } else if (k > number_less + number_equal) {
    parallel::prefix_sum(greater.begin(), greater.end(), greater.begin());
    KASSERT(greater[size - 1] == size-number_equal-number_less, "prefix sum does not work", assert::always);

    StaticArray<T> elements_greater(size - number_equal - number_less);
    tbb::parallel_for(0ul, size, [&](auto i) {
      if (begin[i] > pivot) {
        elements_greater[greater[i] - 1] = begin[i];
      }
    });

    return quickselect_k_smallest<T>(
        k - number_equal - number_less, elements_greater.begin(), elements_greater.end()
    );
  } else {
    return {pivot, number_less, number_equal};
  }
}

} // namespace kaminpar::shm::sparsification::utils
