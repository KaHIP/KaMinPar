/*******************************************************************************
 * Pseudo-clusterer that assigns each node to its own cluster.
 *
 * @file:   noop_clusterer.h
 * @author: Daniel Seemaier
 * @date:   13.05.2022
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/coarsening/clusterer.h"
#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::dist {
class NoopClustering : public Clusterer {
public:
  void cluster(StaticArray<GlobalNodeID> & /* clustering */, const DistributedGraph & /* graph */)
      final {}
};
} // namespace kaminpar::dist
