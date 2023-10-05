/*******************************************************************************
 * Partition metrics for distributed graphs.
 *
 * @file:   metrics.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"

namespace kaminpar::dist::metrics {
/**
 * Computes the number of edges cut in the part of the graph owned by this PE.
 * Includes edges to ghost nodes. Since the graph is directed (there are no
 * reverse edges from ghost nodes to interface nodes), undirected edges are
 * counted twice.
 * @param p_graph Partitioned graph.
 * @return Weighted edge cut of @p p_graph with undirected edges counted twice.
 */
GlobalEdgeWeight local_edge_cut(const DistributedPartitionedGraph &p_graph);

/**
 * Computes the number of edges cut in the whole graph, i.e., across all PEs.
 * Undirected edges are only counted once.
 * @param p_graph Partitioned graph.
 * @return Weighted edge cut across all PEs with undirected edges only counted
 * once.
 */
GlobalEdgeWeight edge_cut(const DistributedPartitionedGraph &p_graph);

/**
 * Computes the partition imbalance of the whole graph partition, i.e., across
 * all PEs. The imbalance of a graph partition is defined as `max_block_weight /
 * avg_block_weight`. Thus, a value of 1.0 indicates that all blocks have the
 * same weight, and a value of 2.0 means that the heaviest block has twice the
 * weight of the average block.
 * @param p_graph Partitioned graph.
 * @return Imbalance of the partition across all PEs.
 */
double imbalance(const DistributedPartitionedGraph &p_graph);

/**
 * Computes whether the blocks of the given partition satisfy the balance
 * constraint given by @p p_ctx.
 * @param p_graph Partitioned graph.
 * @param ctx Partition context describing the balance constraint.
 * @return Whether @p p_graph satisfies the balance constraint given by @p
 * p_ctx.
 */
bool is_feasible(const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx);

/**
 * Counts the number of imbalanced blocks.
 * @param p_graph Partitioned graph.
 * @param p_ctx Partition context describing the maximum block weights.
 * @return The number of imbalanced blocks in the graph.
 */
BlockID
num_imbalanced_blocks(const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx);

/**
 * Computes the L1 distance between the current overloaded block weights and the max block weights.
 * @param p_graph Partitioned graph.
 * @param p_ctx Partition context describing the maximum block weights.
 */
double imbalance_l2(const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx);

/**
 * Computes the L2 distance between the current overloaded block weights and the max block weights.
 * @param p_graph Partitioned graph.
 * @param p_ctx Partition context describing the maximum block weights.
 */
double imbalance_l1(const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx);
} // namespace kaminpar::dist::metrics
