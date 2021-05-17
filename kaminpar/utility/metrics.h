#pragma once

#include "context.h"
#include "datastructure/graph.h"
#include "definitions.h"

#include <numeric>

namespace kaminpar::metrics {
EdgeWeight edge_cut(const PartitionedGraph &p_graph, tag::Parallel);
EdgeWeight edge_cut(const PartitionedGraph &p_graph, tag::Sequential);
inline EdgeWeight edge_cut(const PartitionedGraph &p_graph) { return edge_cut(p_graph, tag::par); }

double imbalance(const PartitionedGraph &p_graph);

NodeWeight overload1(const PartitionedGraph &p_graph, const PartitionContext &context);

double overload2(const PartitionedGraph &p_graph, const PartitionContext &context);

bool is_balanced(const PartitionedGraph &p_graph, const PartitionContext &p_ctx);

bool is_feasible(const PartitionedGraph &p_graph, BlockID input_k, double eps);

bool is_feasible(const PartitionedGraph &p_graph, const PartitionContext &p_ctx);

std::vector<BlockID> find_infeasible_blocks(const PartitionedGraph &p_graph, BlockID input_k, double eps);
} // namespace kaminpar::metrics