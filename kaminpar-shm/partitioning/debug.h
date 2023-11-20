/*******************************************************************************
 * Debug utilities.
 *
 * @file:   debug.h
 * @author: Daniel Seemaier
 * @date:   18.04.2023
 ******************************************************************************/
#pragma once

#include <string>

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"

namespace kaminpar::shm::debug {
void dump_coarsest_graph(const Graph &graph, const Context &ctx);

void dump_graph_hierarchy(const Graph &graph, int level, const Context &ctx);

void dump_graph(const Graph &graph, const std::string &filename);

void dump_coarsest_partition(const PartitionedGraph &p_graph, const Context &ctx);

void dump_partition_hierarchy(
    const PartitionedGraph &p_graph, int level, const std::string &state, const Context &ctx
);

void dump_partition(const PartitionedGraph &p_graph, const std::string &filename);
} // namespace kaminpar::shm::debug
