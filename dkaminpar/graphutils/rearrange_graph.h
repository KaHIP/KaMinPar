/*******************************************************************************
 * @file:   rearrange_graph.h
 *
 * @author: Daniel Seemaier
 * @date:   18.11.2021
 * @brief:  Sort and rearrange a graph by degree buckets.
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"

namespace dkaminpar::graph {
DistributedGraph sort_by_degree_buckets(DistributedGraph graph);
}