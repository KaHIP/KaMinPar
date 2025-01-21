/*******************************************************************************
 * Utility functions for partitioning schemes.
 *
 * @file:   utilities.h
 * @author: Daniel Seemaier
 * @date:   16.01.2023
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::dist {

PartitionContext create_refinement_context(
    const Context &input_ctx, const DistributedGraph &graph, BlockID current_k, bool toplevel
);

shm::PartitionContext create_initial_partitioning_context(
    const Context &input_ctx,
    const shm::Graph &graph,
    BlockID current_block,
    BlockID current_k,
    BlockID desired_k,
    bool toplevel
);

void print_input_graph(const DistributedGraph &graph, bool verbose = false);

void print_coarsened_graph(
    const DistributedGraph &graph,
    const int level,
    GlobalNodeWeight max_cluster_weight,
    bool verbose = false
);

void print_coarsening_converged();

void print_coarsening_terminated(GlobalNodeID desired_num_nodes);

void print_initial_partitioning_result(
    const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
);

} // namespace kaminpar::dist
