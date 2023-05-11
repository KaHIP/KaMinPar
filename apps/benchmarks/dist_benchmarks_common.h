/*******************************************************************************
 * @file:   dist_benchmarks_common.h
 * @author: Daniel Seemaier
 * @date:   25.01.2023
 * @brief:  Common functions for distributed benchmarks.
 ******************************************************************************/
#pragma once

// clang-format off
#include "common/CLI11.h"
#include "dkaminpar/definitions.h"
// clang-format on

#include <string>

#include <tbb/global_control.h>

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist {
tbb::global_control init(const Context &ctx, int &argc, char **&argv);

DistributedGraph load_graph(const std::string &filename);

DistributedPartitionedGraph
load_graph_partition(const DistributedGraph &graph, const std::string &filename);
} // namespace kaminpar::dist
