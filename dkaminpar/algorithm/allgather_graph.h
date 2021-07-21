/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/distributed_definitions.h"
#include "dkaminpar/utility/mpi_helper.h"
#include "kaminpar/datastructure/graph.h"
#include "kaminpar/definitions.h"
#include "kaminpar/utility/metrics.h"

#include <mpi.h>

namespace dkaminpar::graph {
SET_DEBUG(false);

DistributedPartitionedGraph create_from_best_partition(const DistributedGraph &dist_graph,
                                                       shm::PartitionedGraph shm_p_graph,
                                                       MPI_Comm comm = MPI_COMM_WORLD) {
  ALWAYS_ASSERT(dist_graph.global_n() < std::numeric_limits<int>::max()) << "partition size exceeds int size";

  const int rank = mpi::get_comm_rank(comm);
  const shm::EdgeWeight shm_cut = shm::metrics::edge_cut(shm_p_graph);

  // find PE with best partition
  struct ReductionMessage {
    long cut;
    int rank;
  };
  ReductionMessage local{shm_cut, rank};
  ReductionMessage global{};
  MPI_Allreduce(&local, &global, 1, MPI_LONG_INT, MPI_MINLOC, comm);

  // broadcast best partition
  auto partition = shm_p_graph.take_partition();
  MPI_Bcast(partition.data(), static_cast<int>(dist_graph.global_n()), MPI_INT32_T, global.rank, comm);

  // compute block weights
  scalable_vector<shm::parallel::IntegralAtomicWrapper<DBlockWeight>> block_weights(shm_p_graph.k());
  shm_p_graph.pfor_nodes([&](const shm::NodeID u) { block_weights[partition[u]] += shm_p_graph.node_weight(u); });

  // create distributed partition
  scalable_vector<DBlockID> dist_partition(dist_graph.total_n());
  dist_graph.pfor_all_nodes([&](const DNodeID u) { dist_partition[u] = partition[dist_graph.global_node(u)]; });

  // create distributed partitioned graph
  return {&dist_graph, shm_p_graph.k(), std::move(dist_partition), std::move(block_weights)};
}

shm::Graph allgather(const DistributedGraph &graph, MPI_Comm comm = MPI_COMM_WORLD) {
  ALWAYS_ASSERT(graph.n() < std::numeric_limits<int>::max()) << "number of nodes exceeds int size";
  ALWAYS_ASSERT(graph.m() < std::numeric_limits<int>::max()) << "number of edges exceeds int size";

  const DEdgeID m0 = graph.offset_m();

  // TODO can we avoid using two sets of buffers?
  // I.e., create local_x in x?

  // copy own part of the graph to shm datastructures
  shm::StaticArray<shm::EdgeID> local_nodes(graph.n());
  shm::StaticArray<shm::NodeID> local_edges(graph.m());
  shm::StaticArray<shm::NodeWeight> local_node_weights(graph.n());
  shm::StaticArray<shm::EdgeWeight> local_edge_weights(graph.m());

  graph.pfor_nodes([&](const DNodeID u) {
    local_nodes[u] = m0 + static_cast<shm::NodeID>(graph.first_edge(u));
    local_node_weights[u] = static_cast<shm::NodeWeight>(graph.node_weight(u));
    for (const auto [e, v] : graph.neighbors(u)) {
      local_edges[e] = graph.global_node(v);
      local_edge_weights[e] = graph.edge_weight(e);
    }
  });

  // gather graph
  shm::StaticArray<shm::EdgeID> nodes(graph.global_n() + 1);
  shm::StaticArray<shm::NodeID> edges(graph.global_m());
  shm::StaticArray<shm::NodeWeight> node_weights(graph.global_n());
  shm::StaticArray<shm::EdgeWeight> edge_weights(graph.global_m());

  auto nodes_recvcounts = mpi::build_distribution_recvcounts(graph.node_distribution());
  auto nodes_displs = mpi::build_distribution_displs(graph.node_distribution());
  auto edges_recvcounts = mpi::build_distribution_recvcounts(graph.edge_distribution());
  auto edges_displs = mpi::build_distribution_displs(graph.edge_distribution());

  MPI_Allgatherv(local_nodes.data(), static_cast<int>(graph.n()), MPI_UINT32_T, nodes.data(), nodes_recvcounts.data(),
                 nodes_displs.data(), MPI_UINT32_T, comm);
  MPI_Allgatherv(local_node_weights.data(), static_cast<int>(graph.n()), MPI_INT32_T, node_weights.data(),
                 nodes_recvcounts.data(), nodes_displs.data(), MPI_INT32_T, comm);
  MPI_Allgatherv(local_edges.data(), static_cast<int>(graph.m()), MPI_UINT32_T, edges.data(), edges_recvcounts.data(),
                 edges_displs.data(), MPI_UINT32_T, comm);
  MPI_Allgatherv(local_edge_weights.data(), static_cast<int>(graph.m()), MPI_INT32_T, edge_weights.data(),
                 edges_recvcounts.data(), edges_displs.data(), MPI_INT32_T, comm);
  nodes.back() = graph.global_m();

  return {std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights)};
}
} // namespace dkaminpar::graph