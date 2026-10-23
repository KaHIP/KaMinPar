/*******************************************************************************
 * Common preprocessing utilities for cluster contraction implementations.
 *
 * @file:   cluster_contraction_preprocessing.cc
 * @author: Daniel Seemaier
 * @author: Daniel Salwasser
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/coarsening/contraction/cluster_contraction_preprocessing.h"

#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::contraction {

void fill_leader_mapping(
    const Graph &graph,
    const StaticArray<NodeID> &clustering,
    StaticArray<NodeID> &leader_mapping,
    [[maybe_unused]] Timer &timer
) {
  reified(graph, [&](const auto &graph) {
    TIMED_SCOPE_ON(timer, "Allocation") {
      if (leader_mapping.size() < graph.n()) {
        leader_mapping.resize(graph.n(), static_array::noinit);
      }
    };

    TIMED_SCOPE_ON(timer, "Preprocessing") {
      graph.pfor_nodes([&](const NodeID u) { leader_mapping[u] = 0; });
      graph.pfor_nodes([&](const NodeID u) {
        __atomic_store_n(&leader_mapping[clustering[u]], 1, __ATOMIC_RELAXED);
      });
      parallel::prefix_sum(
          leader_mapping.begin(), leader_mapping.begin() + graph.n(), leader_mapping.begin()
      );
    };
  });
}

StaticArray<NodeID> compute_mapping(
    const Graph &graph,
    StaticArray<NodeID> clustering,
    const StaticArray<NodeID> &leader_mapping,
    [[maybe_unused]] Timer &timer
) {
  reified(graph, [&](const auto &graph) {
    TIMED_SCOPE_ON(timer, "Preprocessing") {
      graph.pfor_nodes([&](const NodeID u) {
        clustering[u] = __atomic_load_n(&leader_mapping[clustering[u]], __ATOMIC_RELAXED) - 1;
      });
    };
  });

  return clustering;
}

std::pair<NodeID, StaticArray<NodeID>> compute_mapping(
    const Graph &graph,
    StaticArray<NodeID> clustering,
    MemoryContext &m_ctx,
    [[maybe_unused]] Timer &timer
) {
  SCOPED_HEAP_PROFILER("Compute mapping");

  fill_leader_mapping(graph, clustering, m_ctx.leader_mapping, timer);
  StaticArray<NodeID> mapping =
      compute_mapping(graph, std::move(clustering), m_ctx.leader_mapping, timer);
  const NodeID c_n = m_ctx.leader_mapping[graph.n() - 1];

  TIMED_SCOPE_ON(timer, "Deallocation") {
    m_ctx.leader_mapping.free();
  };

  return {c_n, std::move(mapping)};
}

void fill_cluster_buckets(
    const NodeID c_n,
    const Graph &graph,
    const StaticArray<NodeID> &mapping,
    StaticArray<NodeID> &buckets_index,
    StaticArray<NodeID> &buckets,
    [[maybe_unused]] Timer &timer
) {
  SCOPED_HEAP_PROFILER("Fill cluster buckets");

  TIMED_SCOPE_ON(timer, "Allocation") {
    if (buckets.size() < graph.n()) {
      buckets.resize(graph.n(), static_array::noinit);
    }

    if (buckets_index.size() < c_n + 1) {
      buckets_index.resize(c_n + 1, static_array::noinit);
    }
  };

  TIMED_SCOPE_ON(timer, "Preprocessing") {
    tbb::parallel_for<NodeID>(0, c_n + 1, [&](const NodeID i) { buckets_index[i] = 0; });
    reified(graph, [&](const auto &graph) {
      graph.pfor_nodes([&](const NodeID u) {
        __atomic_fetch_add(&buckets_index[mapping[u]], 1, __ATOMIC_RELAXED);
      });
      parallel::prefix_sum(
          buckets_index.begin(), buckets_index.begin() + c_n + 1, buckets_index.begin()
      );
      graph.pfor_nodes([&](const NodeID u) {
        buckets[__atomic_sub_fetch(&buckets_index[mapping[u]], 1, __ATOMIC_RELAXED)] = u;
      });
    });
  };
}

} // namespace kaminpar::shm::contraction
