/*******************************************************************************
 * Context struct for the distributed graph partitioner.
 *
 * @file:   context.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#pragma once

#include <vector>

#include "kaminpar-dist/datastructures/distributed_graph.h"

#include "kaminpar-shm/datastructures/graph.h"

// Most of the functions are part of the public API
#include "kaminpar-dist/dkaminpar.h"

namespace kaminpar::dist {
struct GraphContext {
public:
  GraphContext(const DistributedGraph &graph, const PartitionContext &p_ctx);
  GraphContext(const shm::Graph &graph, const PartitionContext &p_ctx);

  GlobalNodeID global_n;
  NodeID n;
  NodeID total_n;
  GlobalEdgeID global_m;
  EdgeID m;
  GlobalNodeWeight global_total_node_weight;
  NodeWeight total_node_weight;
  GlobalNodeWeight global_max_node_weight;
  GlobalEdgeWeight global_total_edge_weight;
  EdgeWeight total_edge_weight;

  std::vector<BlockWeight> perfectly_balanced_block_weights;
  std::vector<BlockWeight> max_block_weights;

  [[nodiscard]] inline BlockWeight perfectly_balanced_block_weight(const BlockID b) const {
    return perfectly_balanced_block_weights[b];
  }

  [[nodiscard]] inline BlockWeight max_block_weight(const BlockID b) const {
    return max_block_weights[b];
  }

  void setup_perfectly_balanced_block_weights(BlockID k);
  void setup_max_block_weights(BlockID k, double epsilon);
};
} // namespace kaminpar::dist
