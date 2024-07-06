/*******************************************************************************
 * Utility functions for partitioning.
 *
 * @file:   partition_utils.h
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/abstract_graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm::partitioning {
double compute_2way_adaptive_epsilon(
    NodeWeight total_node_weight, BlockID k, const PartitionContext &p_ctx
);

PartitionContext create_bipartition_context(
    const AbstractGraph &subgraph, BlockID k1, BlockID k2, const PartitionContext &kway_p_ctx
);

/**
 * Given a block $0 <= B < k'$ of an intermediate partition with $k' < k$ blocks, this function
 * computes the number of blocks into which $B$ will be split for the final partition.
 *
 * More precisely, consider a binary tree with labels linked to each node constructed as follows:
 *
 * - The root node has label $k$.
 * - A node with label $\ell > 0$ has two children with labels $\lceil \ell / 2 \rceil$ and $\lfloor
 *   \ell / 2 \rfloor$.
 * - A node with label $\ell = 1$ has one child labelled $1$.
 * - The construction stops as soon as all nodes of a level have label $1$ / the level has size $k$.
 *
 * This function computes the label of any node in this tree, given the size of the nodes level
 * (i.e., $k'$) and its position within the level (i.e., $B$). Note that all levels have distinct
 * sizes, and thus, these two parameters uniquely identify a node of the tree).
 *
 * @param block The block $B$ / the position of a node within its level.
 * @param current_k The number of blocks $k'$ in the intermediate partition / the size of the node's
 * level.
 * @param input_k The number of blocks $k$ in the final partition / the label of the root node.
 *
 * @return The number of blocks into which $B$ will be split for the final partition.
 */
BlockID compute_final_k(BlockID block, BlockID current_k, BlockID input_k);

// compute smallest k_prime such that it is a power of 2 and n / k_prime <= C
BlockID compute_k_for_n(NodeID n, const Context &input_ctx);

std::size_t
compute_num_copies(const Context &input_ctx, NodeID n, bool converged, std::size_t num_threads);

int compute_num_threads_for_parallel_ip(const Context &input_ctx);
} // namespace kaminpar::shm::partitioning
