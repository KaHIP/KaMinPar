/*******************************************************************************
 * Parallel builder for compressed graphs.
 *
 * @file:   parallel_compressed_graph_builder.h
 * @author: Daniel Salwasser
 * @date:   03.05.2024
 ******************************************************************************/
#include "kaminpar-shm/graphutils/parallel_compressed_graph_builder.h"

namespace kaminpar::shm {

CompressedGraph parallel_compress(const CSRGraph &graph) {
  return parallel_compress(
      graph.n(),
      graph.m(),
      graph.is_node_weighted(),
      graph.is_edge_weighted(),
      graph.sorted(),
      [](const NodeID u) { return u; },
      [&](const NodeID u) { return graph.degree(u); },
      [&](const NodeID u) { return graph.first_edge(u); },
      [&](const EdgeID e) { return graph.edge_target(e); },
      [&](const NodeID u) { return graph.node_weight(u); },
      [&](const EdgeID e) { return graph.edge_weight(e); }
  );
}

} // namespace kaminpar::shm