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

#include <tbb/enumerable_thread_specific.h>

namespace dkaminpar::metrics {
DEdgeWeight edge_cut(const DistributedPartitionedGraph &p_graph) {
  tbb::enumerable_thread_specific<DEdgeWeight> cut_ets;

  p_graph.pfor_nodes([&](const DNodeID u) {
    const DBlockID u_block = p_graph.block(u);
    for (const auto [e, v] : p_graph.neighbors(u)) {
      if (u_block != p_graph.block(v)) {
        cut_ets.local() += p_graph.edge_weight(e);
      }
    }
  });

  const DEdgeWeight local_edge_cut = cut_ets.combine(std::plus{});
  DEdgeWeight global_edge_cut;
  MPI_Allreduce(&local_edge_cut, &global_edge_cut, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  ASSERT(global_edge_cut % 2 == 0) << V(global_edge_cut) << V(local_edge_cut);
  return global_edge_cut / 2;
}

double imbalance(const DistributedPartitionedGraph &p_graph) {
    const DNodeWeight local_total_node_weight = p_graph.total_node_weight();
    DNodeWeight global_total_node_weight = 0;
    MPI_Allreduce(&local_total_node_weight, &global_total_node_weight, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

    const double perfect_block_weight = std::ceil(1.0 * global_total_node_weight / p_graph.k());
    double max_imbalance = 0.0;
    for (const DBlockID b : p_graph.blocks()) {
      max_imbalance = std::max(max_imbalance, p_graph.block_weight(b) / perfect_block_weight - 1.0);
    }

    return max_imbalance;
}
}