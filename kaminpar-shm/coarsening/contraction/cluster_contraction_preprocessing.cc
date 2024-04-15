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
namespace {
template <typename Graph>
void fill_leader_mapping(
    const Graph &graph, const StaticArray<NodeID> &clustering, StaticArray<NodeID> &leader_mapping
) {
  START_TIMER("Allocation");
  if (leader_mapping.size() < graph.n()) {
    leader_mapping.resize(graph.n());
  }
  STOP_TIMER();

  RECORD("leader_mapping");
  RECORD_LOCAL_DATA_STRUCT("StaticArray<NodeID>", leader_mapping.size() * sizeof(NodeID));

  START_TIMER("Preprocessing");
  graph.pfor_nodes([&](const NodeID u) { leader_mapping[u] = 0; });
  graph.pfor_nodes([&](const NodeID u) {
    __atomic_store_n(&leader_mapping[clustering[u]], 1, __ATOMIC_RELAXED);
  });
  parallel::prefix_sum(
      leader_mapping.begin(), leader_mapping.begin() + graph.n(), leader_mapping.begin()
  );
  STOP_TIMER();
}

template <typename Graph>
StaticArray<NodeID> compute_mapping(
    const Graph &graph, const StaticArray<NodeID> &clustering, const StaticArray<NodeID> &leader_mapping
) {
  START_TIMER("Allocation");
  RECORD("mapping") StaticArray<NodeID> mapping(graph.n());
  STOP_TIMER();

  START_TIMER("Preprocessing");
  graph.pfor_nodes([&](const NodeID u) {
    mapping[u] = __atomic_load_n(&leader_mapping[clustering[u]], __ATOMIC_RELAXED) - 1;
  });
  STOP_TIMER();

  return mapping;
}

template <typename Graph>
CompactStaticArray<NodeID> compute_compact_mapping(
    const Graph &graph, const StaticArray<NodeID> &clustering, const StaticArray<NodeID> &leader_mapping
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

template <typename Graph, typename Mapping>
void fill_cluster_buckets(
    const NodeID c_n,
    const Graph &graph,
    const Mapping &mapping,
    StaticArray<NodeID> &buckets_index,
    StaticArray<NodeID> &buckets
) {
  START_TIMER("Allocation");
  if (buckets.size() < graph.n()) {
    buckets.resize(graph.n());
  }
  if (buckets_index.size() < c_n + 1) {
    buckets_index.resize(c_n + 1);
  }
  STOP_TIMER();

  RECORD("buckets");
  RECORD_LOCAL_DATA_STRUCT("StaticArray<NodeID>", buckets.size() * sizeof(NodeID));

  RECORD("buckets_index");
  RECORD_LOCAL_DATA_STRUCT("StaticArray<NodeID>", buckets_index.size() * sizeof(NodeID));

  START_TIMER("Preprocessing");
  tbb::parallel_for<std::size_t>(0, buckets_index.size(), [&](const std::size_t i) {
    buckets_index[i] = 0;
  });

  graph.pfor_nodes([&](const NodeID u) {
    __atomic_fetch_add(&buckets_index[mapping[u]], 1, __ATOMIC_RELAXED);
  });

  parallel::prefix_sum(
      buckets_index.begin(), buckets_index.begin() + c_n + 1, buckets_index.begin()
  );
  KASSERT(buckets_index.back() <= graph.n());

  graph.pfor_nodes([&](const NodeID u) {
    buckets[__atomic_sub_fetch(&buckets_index[mapping[u]], 1, __ATOMIC_RELAXED)] = u;
  });

  STOP_TIMER();
}

template <template <typename> typename Mapping, typename Graph>
std::pair<NodeID, Mapping<NodeID>>
generic_preprocess(const Graph &graph, StaticArray<NodeID> &clustering, MemoryContext &m_ctx) {
  auto &buckets_index = m_ctx.buckets_index;
  auto &buckets = m_ctx.buckets;
  auto &leader_mapping = m_ctx.leader_mapping;
  auto &all_buffered_nodes = m_ctx.all_buffered_nodes;

  fill_leader_mapping(graph, clustering, leader_mapping);
  Mapping<NodeID> mapping;
  if constexpr (std::is_same_v<Mapping<NodeID>, StaticArray<NodeID>>) {
    mapping = compute_mapping(graph, clustering, leader_mapping);
  } else {
    mapping = compute_compact_mapping(graph, clustering, leader_mapping);
  }

  const NodeID c_n = leader_mapping[graph.n() - 1];

  TIMED_SCOPE("Allocation") {
    leader_mapping.free();
    clustering.free();
  };

  fill_cluster_buckets(c_n, graph, mapping, buckets_index, buckets);

  return {c_n, std::move(mapping)};
}
} // namespace

template <template <typename> typename Mapping>
std::pair<NodeID, Mapping<NodeID>>
preprocess(const Graph &graph, StaticArray<NodeID> &clustering, MemoryContext &m_ctx) {
  return graph.reified([&](auto &graph) {
    return generic_preprocess<Mapping>(graph, clustering, m_ctx);
  });
}

template std::pair<NodeID, StaticArray<NodeID>>
preprocess<StaticArray>(const Graph &graph, StaticArray<NodeID> &clustering, MemoryContext &m_ctx);

template std::pair<NodeID, CompactStaticArray<NodeID>> preprocess<CompactStaticArray>(
    const Graph &graph, StaticArray<NodeID> &clustering, MemoryContext &m_ctx
);
} // namespace kaminpar::shm::contraction
