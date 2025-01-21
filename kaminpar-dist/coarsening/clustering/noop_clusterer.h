/*******************************************************************************
 * Pseudo-clusterer that assigns each node to its own cluster.
 *
 * @file:   noop_clusterer.h
 * @author: Daniel Seemaier
 * @date:   13.05.2022
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/coarsening/clusterer.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::dist {

class NoopClustering : public Clusterer {
public:
  NoopClustering(const bool local_clusterer) : _local_clusterer(local_clusterer) {}

  void cluster(StaticArray<GlobalNodeID> &clustering, const DistributedGraph &graph) final {
    if (_local_clusterer) {
      StaticArray<NodeID> local_clustering(
          graph.n(), reinterpret_cast<NodeID *>(clustering.data())
      );
      graph.pfor_nodes([&](const NodeID node) { local_clustering[node] = node; });
    } else {
      graph.pfor_all_nodes([&](const NodeID node) {
        clustering[node] = graph.local_to_global_node(node);
      });
    }
  }

private:
  bool _local_clusterer = false;
};

} // namespace kaminpar::dist
