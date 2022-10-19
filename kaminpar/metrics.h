/*******************************************************************************
 * @file:   metrics.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Functions to compute partition quality metric.s
 ******************************************************************************/
#pragma once

#include <numeric>

#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/definitions.h"

namespace kaminpar::shm::metrics {
EdgeWeight        edge_cut(const PartitionedGraph& p_graph, tag::Parallel);
EdgeWeight        edge_cut(const PartitionedGraph& p_graph, tag::Sequential);
inline EdgeWeight edge_cut(const PartitionedGraph& p_graph) {
    return edge_cut(p_graph, tag::par);
}

double imbalance(const PartitionedGraph& p_graph);

NodeWeight total_overload(const PartitionedGraph& p_graph, const PartitionContext& context);

bool is_balanced(const PartitionedGraph& p_graph, const PartitionContext& p_ctx);

bool is_feasible(const PartitionedGraph& p_graph, BlockID input_k, double eps);

bool is_feasible(const PartitionedGraph& p_graph, const PartitionContext& p_ctx);
} // namespace kaminpar::shm::metrics
