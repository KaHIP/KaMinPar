/*******************************************************************************
 * Implements common synchronization operations for distributed graphs.
 *
 * @file:   graph_synchronization.h
 * @author: Daniel Seemaier
 * @date:   15.07.2022
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/graphutils/communication.h"

namespace kaminpar::dist::graph {
/*!
 * Synchronizes the block assignment of ghost nodes: each node sends its current
 * assignment to all replicates (ghost nodes) residing on other PEs.
 *
 * @param p_graph Graph partition to synchronize.
 */
void synchronize_ghost_node_block_ids(DistributedPartitionedGraph &p_graph);

template <typename Graph> void synchronize_ghost_node_weights(Graph &graph) {
  struct Message {
    NodeID node;
    NodeWeight weight;
  };

  mpi::graph::sparse_alltoall_interface_to_pe<Message>(
      graph,
      [&](const NodeID u) -> Message { return {.node = u, .weight = graph.node_weight(u)}; },
      [&](const auto &recv_buffer, const PEID pe) {
        tbb::parallel_for<std::size_t>(0, recv_buffer.size(), [&](const std::size_t i) {
          const auto [local_node_on_pe, weight] = recv_buffer[i];
          const auto global_node = static_cast<GlobalNodeID>(graph.offset_n(pe) + local_node_on_pe);
          const NodeID local_node = graph.global_to_local_node(global_node);
          graph.set_ghost_node_weight(local_node, weight);
        });
      }
  );
}
} // namespace kaminpar::dist::graph
