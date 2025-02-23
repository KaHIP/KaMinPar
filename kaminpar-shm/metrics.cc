/*******************************************************************************
 * Utility functions for computing partition metrics.
 *
 * @file:   metrics.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/metrics.h"

#include <cstdint>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/asserting_cast.h"

namespace kaminpar::shm::metrics {

namespace {

template <typename PartitionedGraph, typename Graph>
EdgeWeight edge_cut_seq(const PartitionedGraph &p_graph, const Graph &graph) {
  std::int64_t cut = 0;

  for (const NodeID u : graph.nodes()) {
    graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      cut += (p_graph.block(u) != p_graph.block(v)) ? w : 0;
    });
  }

  KASSERT(cut % 2 == 0u, "inconsistent cut", assert::always);
  return asserting_cast<assert::always, EdgeWeight>(cut / 2);
}

template <typename PartitionedGraph, typename Graph>
EdgeWeight edge_cut(const PartitionedGraph &p_graph, const Graph &graph) {
  tbb::enumerable_thread_specific<int64_t> cut_ets;

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, graph.n()), [&](const auto &r) {
    auto &cut = cut_ets.local();
    for (NodeID u = r.begin(); u < r.end(); ++u) {
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        cut += (p_graph.block(u) != p_graph.block(v)) ? w : 0;
      });
    }
  });

  std::int64_t cut = cut_ets.combine(std::plus<>{});

  KASSERT(cut % 2 == 0u, "inconsistent cut", assert::always);
  return asserting_cast<assert::always, EdgeWeight>(cut / 2);
}

} // namespace

EdgeWeight edge_cut(const PartitionedGraph &p_graph) {
  return reified(p_graph, [&](const auto &graph) { return edge_cut(p_graph, graph); });
}

EdgeWeight edge_cut_seq(const PartitionedGraph &p_graph) {
  return reified(p_graph, [&](const auto &graph) { return edge_cut_seq(p_graph, graph); });
}

EdgeWeight edge_cut_seq(const PartitionedCSRGraph &p_graph) {
  return edge_cut_seq(p_graph, p_graph.graph());
}

EdgeWeight edge_cut(const PartitionedCSRGraph &p_graph) {
  return edge_cut(p_graph, p_graph.graph());
}

} // namespace kaminpar::shm::metrics
