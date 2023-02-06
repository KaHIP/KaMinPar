/*******************************************************************************
 * @file:   debug.h
 * @author: Daniel Seemaier
 * @date:   16.05.2022
 * @brief:  Debug features.
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"

#include "kaminpar/datastructures/graph.h"

#include "common/parallel/atomic.h"

namespace kaminpar::dist::debug {
void save_partition(const DistributedPartitionedGraph& p_graph, const Context& ctx, int level);

void save_graph(const DistributedGraph& graph, const Context& ctx, int level);

void save_graph(const shm::Graph& graph, const Context& ctx, const int level);

void save_partitioned_graph(const DistributedPartitionedGraph& p_graph, const Context& ctx, int level);

void save_global_clustering(
    const scalable_vector<parallel::Atomic<GlobalNodeID>>& mapping, const Context& ctx, int level
);
} // namespace kaminpar::dist::debug
