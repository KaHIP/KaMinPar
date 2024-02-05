/*******************************************************************************
 * Utility functions for partitioning schemes.
 *
 * @file:   utilities.h
 * @author: Daniel Seemaier
 * @date:   16.01.2023
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"

namespace kaminpar::dist {
void print_input_graph(const DistributedGraph &graph);

void print_coarsened_graph(
    const DistributedGraph &graph, const int level, GlobalNodeWeight max_cluster_weight
);

void print_coarsening_converged();

void print_coarsening_terminated(GlobalNodeID desired_num_nodes);

void print_initial_partitioning_result(
    const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
);
} // namespace kaminpar::dist
