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
#include "dkaminpar/distributed_context.h"
#include "dkaminpar/distributed_definitions.h"

#include <tbb/enumerable_thread_specific.h>

namespace dkaminpar::metrics {
/*!
 * Computes the number of edges cut in the part of the graph owned by this PE. Includes edges to ghost nodes. Since the
 * graph is directed (there are no reverse edges from ghost nodes to interface nodes), undirected edges are counted
 * twice.
 * @param p_graph Partitioned graph.
 * @return Weighted edge cut of @p p_graph with undirected edges counted twice.
 */
EdgeWeight local_edge_cut(const DistributedPartitionedGraph &p_graph);

/*!
 * Computes the number of edges cut in the whole graph, i.e., across all PEs. Undirected edges are only counted once.
 * @param p_graph Partitioned graph.
 * @return Weighted edge cut across all PEs with undirected edges only counted once.
 */
GlobalEdgeWeight edge_cut(const DistributedPartitionedGraph &p_graph);

/*!
 * Computes the partition imbalance of the whole graph partition, i.e., across all PEs.
 * The imbalance of a graph partition is defined as `max_block_weight / avg_block_weight`. Thus, a value of 1.0
 * indicates that all blocks have the same weight, and a value of 2.0 means that the heaviest block has twice the
 * weight of the average block.
 * @param p_graph Partitioned graph.
 * @return Imbalance of the partition across all PEs.
 */
double imbalance(const DistributedPartitionedGraph &p_graph);

/*!
 * Computes whether the blocks of the given partition satisfy the balance constraint given by @p p_ctx.
 * @param p_graph Partitioned graph.
 * @param ctx Partition context describing the balance constraint.
 * @return Whether @p p_graph satisfies the balance constraint given by @p p_ctx.
 */
bool is_feasible(const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx);
} // namespace dkaminpar::metrics