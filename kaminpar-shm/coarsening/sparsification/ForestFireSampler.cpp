//
// Created by badger on 5/8/24.
//

#include "ForestFireSampler.h"

#include <networkit/graph/Graph.hpp>
#include <networkit/include/networkit/sparsification/ForestFireScore.hpp>

#include "networkit_utils.h"
#include "sparsification_utils.h"

#include "kaminpar-common/timer.h"

namespace kaminpar::shm::sparsification {
StaticArray<EdgeWeight> ForestFireSampler::sample(const CSRGraph &g, EdgeID target_edge_amount) {
  auto begin = std::chrono::high_resolution_clock::now();
  NetworKit::Graph nk_graph = networkit_utils::toNetworKitGraph(g);
  nk_graph.indexEdges();
  nk_graph.sortEdges();
  printf(
      "* Time to create NetworKit graph: %ld\n",
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - begin
      )
          .count()
  );

  begin = std::chrono::high_resolution_clock::now();
  NetworKit::ForestFireScore forest_fire_score(nk_graph, _pf, _targetBurntRatio);
  forest_fire_score.run();
  auto scores = forest_fire_score.scores();
  printf(
      "* Time to run ForestFireScore: %ld\n",
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - begin
      )
          .count()
  );

  begin = std::chrono::high_resolution_clock::now();
  auto sorted_scores = scores;
  std::sort(sorted_scores.begin(), sorted_scores.end(), std::greater<>());
  // divide by 2, because, sores only has an entry for evey undirected edge in the graph,
  // while target edge amount counts edges in a directed way
  EdgeID threshold = sorted_scores[target_edge_amount / 2];

  auto sorted_by_target = utils::sort_by_traget(g);
  auto sample = StaticArray<EdgeWeight>(g.m());
  for (NodeID u : g.nodes()) {
    for (EdgeID i = 0; i < g.degree(u); i++) {
      EdgeID e = sorted_by_target[g.raw_nodes()[u] + i];
      auto [v, nk_e] = nk_graph.getIthNeighborWithId(u, i);
      KASSERT(g.edge_target(e) == v, "edge target does not match");
      sample[e] = scores[nk_e] > threshold ? g.edge_weight(e) : 0;
    }
  }
  printf(
      "* Time to create sample: %ld\n",
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::high_resolution_clock::now() - begin
      )
          .count()
  );

  return sample;
}
} // namespace kaminpar::shm::sparsification