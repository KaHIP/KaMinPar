#pragma once

#include <functional>

#include <oneapi/tbb/concurrent_vector.h>
#include <oneapi/tbb/parallel_sort.h>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
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
    g.pfor_neighbors(u, g.n() - 1, 2000, [&](EdgeID e, NodeID v, EdgeWeight e_weight) {
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

template <typename T, typename Iterator> T medians_of_medians(Iterator begin, Iterator end);

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

template <typename T, typename Iterator>
T quickselect_k_smallest(size_t k, Iterator begin, Iterator end) {
  size_t size = std::distance(begin, end);
  if (size <= 20)
    return sortselect_k_smallest<T, Iterator>(k, begin, end);
  T pivot = medians_of_medians<T, Iterator>(begin, end);
  tbb::concurrent_vector<T> less = {}, greater = {};
  tbb::parallel_for(0ul, size, [&](size_t i) {
    T x = begin[i];
    if (x <= pivot)
      less.push_back(x);
    else
      greater.push_back(x);
  });

  KASSERT(
      less.size() <= 0.7 * size + 2 && greater.size() <= 0.7 * size + 2,
      "median of medians privot guarantee does not hold",
      assert::always
  );
  if (k <= less.size())
    return quickselect_k_smallest<T, typename tbb::concurrent_vector<T>::iterator>(
        k, less.begin(), less.end()
    );
  else
    return quickselect_k_smallest<T, typename tbb::concurrent_vector<T>::iterator>(
        k - less.size(), greater.begin(), greater.end()
    );
}

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

template <typename T, typename Iterator> T medians_of_medians(Iterator begin, Iterator end) {
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
  );
}

template <typename WeightIterator>
StaticArray<size_t>
sample_k_without_replacement(WeightIterator weights_begin, WeightIterator weights_end, size_t k) {
  StaticArray<double> test(2);
  auto size = weights_end - weights_begin;
  StaticArray<double> keys(size);
  tbb::parallel_for(0ul, size, [&](auto i) {
    keys[i] = -std::log(Random::instance().random_double()) / weights_begin[i];
  });
  double x = quickselect_k_smallest<double>(k, keys.begin(), keys.end());

  tbb::concurrent_vector<double> selected;
  size_t back = 0;
  tbb::parallel_for(0ul, keys.size(), [&](auto i) {
    if (keys[i] <= x) {
      __atomic_fetch_add(&back, 1, __ATOMIC_RELAXED);
      selected.push_back(i);
    }
  });
  return StaticArray<size_t>(selected.begin(), selected.end());
}

} // namespace kaminpar::shm::sparsification::utils
