//
// Created by badger on 5/8/24.
//

#include "ForestFireSampler.h"

#include <networkit/graph/Graph.hpp>
#include <networkit/include/networkit/sparsification/ForestFireScore.hpp>

#include "networkit_utils.h"

namespace kaminpar::shm::sparsification {
StaticArray<EdgeWeight> ForestFireSampler::sample(const CSRGraph &g) {
  NetworKit::Graph nk_graph = networkit_utils::toNetworKitGraph(g);
  nk_graph.indexEdges();

  NetworKit::ForestFireScore forest_fire_score(nk_graph, _pf, _targetBurntRatio);
  forest_fire_score.run();
  auto scores = forest_fire_score.scores();

  auto sample = StaticArray<EdgeWeight>(g.m());
  for (NodeID u : g.nodes()) {
    for (EdgeID e : g.incident_edges(u)) {
      sample[e] = scores[nk_graph.edgeId(u, g.edge_target(e))] >= _threshold ? g.edge_weight(e) : 0;
    }
  }

  return sample;
}
} // namespace kaminpar::shm::sparsification