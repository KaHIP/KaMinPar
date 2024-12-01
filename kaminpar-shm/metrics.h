/*******************************************************************************
 * Utility functions for computing partition metrics.
 *
 * @file:   metrics.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <cmath>
#include <cstdint>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/asserting_cast.h"

namespace kaminpar::shm::metrics {

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

template <typename PartitionedGraph> EdgeWeight edge_cut(const PartitionedGraph &p_graph) {
  return p_graph.reified([&](const auto &graph) { return edge_cut(p_graph, graph); });
}

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

template <typename PartitionedGraph> EdgeWeight edge_cut_seq(const PartitionedGraph &p_graph) {
  return p_graph.reified([&](const auto &graph) { return edge_cut_seq(p_graph, graph); });
}

template <typename PartitionedGraph> double imbalance(const PartitionedGraph &p_graph) {
  const double perfect_block_weight = std::ceil(1.0 * p_graph.total_node_weight() / p_graph.k());

  double max_imbalance = 0.0;
  for (const BlockID b : p_graph.blocks()) {
    max_imbalance = std::max(max_imbalance, p_graph.block_weight(b) / perfect_block_weight - 1.0);
  }

  return max_imbalance;
}

template <typename PartitionedGraph>
NodeWeight total_overload(const PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  NodeWeight total_overload = 0;

  for (const BlockID b : p_graph.blocks()) {
    total_overload += std::max<BlockWeight>(0, p_graph.block_weight(b) - p_ctx.max_block_weight(b));
  }

  return total_overload;
}

template <typename PartitionedGraph>
bool is_balanced(const PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  return std::all_of(p_graph.blocks().begin(), p_graph.blocks().end(), [&](const BlockID b) {
    return p_graph.block_weight(b) <= p_ctx.max_block_weight(b);
  });
}

template <typename PartitionedGraph>
bool is_feasible(const PartitionedGraph &p_graph, const BlockID input_k, const double eps) {
  const double max_block_weight = std::ceil((1.0 + eps) * p_graph.total_node_weight() / input_k);

  return std::all_of(p_graph.blocks().begin(), p_graph.blocks().end(), [&](const BlockID b) {
    const BlockID final_kb = compute_final_k(b, p_graph.k(), input_k);
    return p_graph.block_weight(b) <= max_block_weight * final_kb + p_graph.max_node_weight();
  });
}

template <typename PartitionedGraph>
bool is_feasible(const PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  return is_balanced(p_graph, p_ctx);
}

} // namespace kaminpar::shm::metrics
