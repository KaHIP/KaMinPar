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
#include "dkaminpar/utility/distributed_metrics.h"

#include "dkaminpar/mpi_wrapper.h"

namespace dkaminpar::metrics {
EdgeWeight local_edge_cut(const DistributedPartitionedGraph &p_graph) {
  tbb::enumerable_thread_specific<EdgeWeight> cut_ets;

  p_graph.pfor_nodes_range([&](const auto r) {
    auto &cut = cut_ets.local();
    for (NodeID u = r.begin(); u < r.end(); ++u) {
      const BlockID u_block = p_graph.block(u);
      for (const auto [e, v] : p_graph.neighbors(u)) {
        if (u_block != p_graph.block(v)) { cut += p_graph.edge_weight(e); }
      }
    }
  });

  return cut_ets.combine(std::plus{});
}

GlobalEdgeWeight edge_cut(const DistributedPartitionedGraph &p_graph) {
  const GlobalEdgeWeight global_edge_cut = mpi::allreduce(static_cast<GlobalEdgeWeight>(local_edge_cut(p_graph)),
                                                          MPI_SUM, p_graph.communicator());
  ASSERT(global_edge_cut % 2 == 0);
  return global_edge_cut / 2;
}

double imbalance(const DistributedPartitionedGraph &p_graph) {
  const auto global_total_node_weight = mpi::allreduce<GlobalNodeWeight>(p_graph.total_node_weight(), MPI_SUM,
                                                                         p_graph.communicator());

  const double perfect_block_weight = std::ceil(static_cast<double>(global_total_node_weight) / p_graph.k());
  double max_imbalance = 0.0;
  for (const BlockID b : p_graph.blocks()) {
    max_imbalance = std::max(max_imbalance, static_cast<double>(p_graph.block_weight(b)) / perfect_block_weight - 1.0);
  }

  return max_imbalance;
}

bool is_feasible(const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  return imbalance(p_graph) < p_ctx.epsilon;
}
} // namespace dkaminpar::metrics