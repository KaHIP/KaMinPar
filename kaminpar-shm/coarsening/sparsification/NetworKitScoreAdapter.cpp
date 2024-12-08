//
// Created by badger on 6/12/24.
//

#include "NetworKitScoreAdapter.h"

#include <networkit/graph/Graph.hpp>
#include <networkit/sparsification/ForestFireScore.hpp>

#include "NetworKitWeightedForestFireScore.hpp"
#include "networkit_utils.h"
#include "sparsification_utils.h"

namespace kaminpar::shm::sparsification {
template <typename EdgeScore, typename Score>
StaticArray<Score> NetworKitScoreAdapter<EdgeScore, Score>::scores(const CSRGraph &g) {
  NetworKit::Graph *nk_graph = networkit_utils::toNetworKitGraph(g);
  nk_graph->indexEdges();
  nk_graph->sortEdges();

  auto scorer = _curried_constructor(*nk_graph);
  scorer.run();
  auto nk_scores = scorer.scores();

  auto sorted_by_target = utils::sort_by_traget(g);
  auto scores = StaticArray<Score>(g.m());
  for (NodeID u : g.nodes()) {
    for (EdgeID i = 0; i < g.degree(u); i++) {
      EdgeID e = sorted_by_target[g.raw_nodes()[u] + i];
      auto [v, nk_e] = nk_graph->getIthNeighborWithId(u, i);
      KASSERT(g.edge_target(e) == v, "edge target does not match");
      scores[e] = nk_scores[nk_e];
    }
  }

  delete (nk_graph);

  return scores;
}
template class NetworKitScoreAdapter<NetworKit::ForestFireScore, double>;
template class NetworKitScoreAdapter<NetworKitWeightedForestFireScore, double>;

} // namespace kaminpar::shm::sparsification
