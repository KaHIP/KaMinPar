/*******************************************************************************
 * Common preprocessing utilities for cluster contraction implementations.
 *
 * @file:   cluster_contraction_preprocessing.cc
 * @author: Daniel Seemaier
 * @author: Daniel Salwasser
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/coarsening/contraction/cluster_contraction_preprocessing.h"

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/compact_static_array.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::contraction {
void fill_leader_mapping(
    const Graph &graph, const StaticArray<NodeID> &clustering, StaticArray<NodeID> &leader_mapping
) {
  TIMED_SCOPE("Allocation") {
    if (leader_mapping.size() < graph.n()) {
      RECORD("leader_mapping") leader_mapping.resize(graph.n(), static_array::noinit);
      RECORD_LOCAL_DATA_STRUCT("StaticArray<NodeID>", leader_mapping.size() * sizeof(NodeID));
    }
  };

  TIMED_SCOPE("Preprocessing") {
    graph.pfor_nodes([&](const NodeID u) { leader_mapping[u] = 0; });
    graph.pfor_nodes([&](const NodeID u) {
      __atomic_store_n(&leader_mapping[clustering[u]], 1, __ATOMIC_RELAXED);
    });
    parallel::prefix_sum(
        leader_mapping.begin(), leader_mapping.begin() + graph.n(), leader_mapping.begin()
    );
  };
}

template <>
StaticArray<NodeID> compute_mapping(
    const Graph &graph, StaticArray<NodeID> clustering, const StaticArray<NodeID> &leader_mapping
) {
  START_TIMER("Preprocessing");
  graph.pfor_nodes([&](const NodeID u) {
    clustering[u] = __atomic_load_n(&leader_mapping[clustering[u]], __ATOMIC_RELAXED) - 1;
  });
  STOP_TIMER();

  return std::move(clustering);
}

template <>
CompactStaticArray<NodeID> compute_mapping(
    const Graph &graph, StaticArray<NodeID> clustering, const StaticArray<NodeID> &leader_mapping
) {
  const NodeID c_n = leader_mapping[graph.n() - 1];

  START_TIMER("Allocation");
  RECORD("mapping") CompactStaticArray<NodeID> mapping(math::byte_width(c_n), graph.n());
  STOP_TIMER();

  START_TIMER("Preprocessing");
  graph.pfor_nodes([&](const NodeID u) {
    mapping.write(u, __atomic_load_n(&leader_mapping[clustering[u]], __ATOMIC_RELAXED) - 1);
  });
  STOP_TIMER();

  return mapping;
}

template <template <typename> typename Mapping>
std::pair<NodeID, Mapping<NodeID>>
compute_mapping(const Graph &graph, StaticArray<NodeID> clustering, MemoryContext &m_ctx) {
  SCOPED_HEAP_PROFILER("Compute mapping");

  fill_leader_mapping(graph, clustering, m_ctx.leader_mapping);
  Mapping<NodeID> mapping =
      compute_mapping<Mapping>(graph, std::move(clustering), m_ctx.leader_mapping);
  const NodeID c_n = m_ctx.leader_mapping[graph.n() - 1];

  TIMED_SCOPE("Deallocation") {
    m_ctx.leader_mapping.free();
  };

  return {c_n, std::move(mapping)};
}

template std::pair<NodeID, StaticArray<NodeID>> compute_mapping<StaticArray>(
    const Graph &graph, StaticArray<NodeID> clustering, MemoryContext &m_ctx
);

template std::pair<NodeID, CompactStaticArray<NodeID>> compute_mapping<CompactStaticArray>(
    const Graph &graph, StaticArray<NodeID> clustering, MemoryContext &m_ctx
);

template <typename Mapping>
void fill_cluster_buckets(
    const NodeID c_n,
    const Graph &graph,
    const Mapping &mapping,
    StaticArray<NodeID> &buckets_index,
    StaticArray<NodeID> &buckets
) {
  SCOPED_HEAP_PROFILER("Fill cluster buckets");

  TIMED_SCOPE("Allocation") {
    if (buckets.size() < graph.n()) {
      RECORD("buckets") buckets.resize(graph.n(), static_array::noinit);
      RECORD_LOCAL_DATA_STRUCT("StaticArray<NodeID>", buckets.size() * sizeof(NodeID));
    }

    if (buckets_index.size() < c_n + 1) {
      RECORD("buckets_index") buckets_index.resize(c_n + 1, static_array::noinit);
      RECORD_LOCAL_DATA_STRUCT("StaticArray<NodeID>", buckets_index.size() * sizeof(NodeID));
    }
  };

  TIMED_SCOPE("Preprocessing") {
    tbb::parallel_for<NodeID>(0, c_n + 1, [&](const NodeID i) { buckets_index[i] = 0; });
    graph.pfor_nodes([&](const NodeID u) {
      __atomic_fetch_add(&buckets_index[mapping[u]], 1, __ATOMIC_RELAXED);
    });
    parallel::prefix_sum(
        buckets_index.begin(), buckets_index.begin() + c_n + 1, buckets_index.begin()
    );
    graph.pfor_nodes([&](const NodeID u) {
      buckets[__atomic_sub_fetch(&buckets_index[mapping[u]], 1, __ATOMIC_RELAXED)] = u;
    });
  };
}

template void fill_cluster_buckets(
    NodeID c_n,
    const Graph &graph,
    const StaticArray<NodeID> &mapping,
    StaticArray<NodeID> &buckets_index,
    StaticArray<NodeID> &buckets
);

template void fill_cluster_buckets(
    NodeID c_n,
    const Graph &graph,
    const CompactStaticArray<NodeID> &mapping,
    StaticArray<NodeID> &buckets_index,
    StaticArray<NodeID> &buckets
);
} // namespace kaminpar::shm::contraction
