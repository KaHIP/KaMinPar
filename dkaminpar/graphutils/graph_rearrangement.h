/*******************************************************************************
 * @file:   graph_rearrangement.h
 * @author: Daniel Seemaier
 * @date:   18.11.2021
 * @brief:  Sort and rearrange a graph by degree buckets.
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist::graph {
DistributedGraph sort_by_degree_buckets(DistributedGraph graph);
}
