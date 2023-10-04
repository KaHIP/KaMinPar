/*******************************************************************************
 * Finds all border nodes of a partitioned graph.
 *
 * @file:   border_nodes.cc
 * @author: Daniel Seemaier
 * @date:   20.09.2023
 ******************************************************************************/
#include "kaminpar-dist/algorithms/border_nodes.h"

#include <vector>

#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-common/timer.h"

namespace kaminpar::dist::graph {
std::vector<NodeID> find_border_nodes(const DistributedPartitionedGraph &p_graph) {
  std::vector<NodeID> border_nodes;

  for (const NodeID u : p_graph.nodes()) {
    const BlockID bu = p_graph.block(u);
    for (const auto [e, v] : p_graph.neighbors(u)) {
      if (p_graph.block(v) != bu) {
        border_nodes.push_back(u);
        break;
      }
    }
  }

  return border_nodes;
}
} // namespace kaminpar::dist::graph
