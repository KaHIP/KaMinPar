/*******************************************************************************
 * @file:   debug.h
 * @author: Daniel Seemaier
 * @date:   16.05.2022
 * @brief:  Debug features.
 ******************************************************************************/
#pragma once

#include "dkaminpar/coarsening/global_clustering_contraction.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructure/distributed_graph.h"

namespace kaminpar::dist::debug {
void save_partition(const DistributedPartitionedGraph& p_graph, const Context& ctx, const int level);

void save_graph(const DistributedGraph& graph, const Context& ctx, const int level);

void save_partitioned_graph(const DistributedPartitionedGraph& p_graph, const Context& ctx, const int level);

void save_global_clustering(const scalable_vector<Atomic<GlobalNodeID>>& mapping, const Context& ctx, const int level);
} // namespace kaminpar::dist::debug
