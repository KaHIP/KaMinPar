/*******************************************************************************
 * @file:   rearrange_graph.h
 *
 * @author: Daniel Seemaier
 * @date:   18.11.2021
 * @brief:  Sort and rearrange a graph by degree buckets.
 ******************************************************************************/
#include "dkaminpar/graphutils/rearrange_graph.h"

#include "dkaminpar/mpi_graph.h"
#include "kaminpar/datastructure/marker.h"
#include "kaminpar/graphutils/graph_rearrangement.h"
#include "kaminpar/parallel.h"
#include "kaminpar/utility/timer.h"

namespace dkaminpar::graph {
DistributedGraph sort_by_degree_buckets(DistributedGraph graph) {
  SCOPED_TIMER("Sort and rearrange graph");

  const auto &old_nodes = graph.raw_nodes();
  const auto &old_edges = graph.raw_edges();
  const auto &old_node_weights = graph.raw_node_weights();
  const auto &old_edge_weights = graph.raw_edge_weights();

  auto permutations = shm::graph::sort_by_degree_buckets<scalable_vector>(old_nodes);

  // rearrange nodes, edges, node weights and edge weights
  // ghost nodes are copied without remapping them to new IDs
  START_TIMER("Allocation");
  scalable_vector<EdgeID> new_nodes(old_nodes.size());
  scalable_vector<NodeID> new_edges(old_edges.size());
  scalable_vector<NodeWeight> new_node_weights(old_node_weights.size());
  scalable_vector<EdgeWeight> new_edge_weights(old_edge_weights.size());
  STOP_TIMER();
  shm::graph::build_permuted_graph<scalable_vector, true>(old_nodes, old_edges, old_node_weights, old_edge_weights,
                                                          permutations, new_nodes, new_edges, new_node_weights,
                                                          new_edge_weights);

  // communicate new global IDs of ghost nodes
  struct ChangedNodeLabel {
    NodeID old_node_local;
    NodeID new_node_local;
  };

  auto received =
      mpi::graph::sparse_alltoall_interface_to_pe_get<ChangedNodeLabel>(graph, [&](const NodeID u) -> ChangedNodeLabel {
        return {.old_node_local = u, .new_node_local = permutations.old_to_new[u]};
      });

  auto old_global_to_ghost = graph.take_global_to_ghost(); // TODO cannot be cleared?
  growt::StaticGhostNodeMapping new_global_to_ghost(old_global_to_ghost.capacity());
  auto new_ghost_to_global = graph.take_ghost_to_global(); // can be reused

  shm::parallel::chunked_for(received, [&](const ChangedNodeLabel &message, const PEID pe) {
    const auto &[old_node_local, new_node_local] = message;
    const GlobalNodeID old_node_global = graph.offset_n(pe) + old_node_local;
    const GlobalNodeID new_node_global = graph.offset_n(pe) + new_node_local;

    const NodeID ghost_node = (*old_global_to_ghost.find(old_node_global)).second;
    new_global_to_ghost.insert(new_node_global, ghost_node);
    new_ghost_to_global[ghost_node] = new_node_global;
  });

  return {graph.take_node_distribution(), graph.take_edge_distribution(),
          std::move(new_nodes),           std::move(new_edges),
          std::move(new_node_weights),    std::move(new_edge_weights),
          graph.take_ghost_owner(),       std::move(new_ghost_to_global),
          std::move(new_global_to_ghost), graph.communicator()};
}
} // namespace dkaminpar::graph