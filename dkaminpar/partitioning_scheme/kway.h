/*******************************************************************************
* @file:   kway.h
*
* @author: Daniel Seemaier
* @date:   25.10.2021
* @brief:  Direct k-way partitioning.
******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/context.h"

namespace dkaminpar {
class KWayPartitioningScheme {
public:
  KWayPartitioningScheme(const DistributedGraph &graph, const Context &ctx);

  DistributedPartitionedGraph partition();

private:
  const DistributedGraph &_graph;
  const Context &_ctx;
};
} // namespace dkaminpar