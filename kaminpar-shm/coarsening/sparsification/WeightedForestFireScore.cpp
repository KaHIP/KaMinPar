#include "WeightedForestFireScore.h"

#include <functional>
#include <queue>

#include <oneapi/tbb/concurrent_vector.h>

#include "DistributionDecorator.h"
#include "IndexDistributionWithoutReplacement.h"

#include "kaminpar-common/random.h"

namespace kaminpar::shm::sparsification {
StaticArray<EdgeID> WeightedForestFireScore::scores(const CSRGraph &g) {
  std::vector<std::atomic_uint32_t> burnt(g.m());
  tbb::parallel_for(0ul, burnt.size(), [&](auto i) { burnt[i] = 0; });
  std::atomic_uint32_t edges_burnt = 0;

  while (edges_burnt < _targetBurnRatio * g.m()) {

    // Start a new fire
    std::queue<NodeID> activeNodes;
    StaticArray<std::atomic_bool> visited(g.m(), false);
    activeNodes.push(Random::instance().random_index(0, g.n()));

    auto forwardNeighborDistribution = [&](NodeID u) {
      tbb::concurrent_vector<EdgeID> validEdges;
      tbb::concurrent_vector<EdgeWeight> weights;
      tbb::parallel_for(*g.incident_edges(u).begin(), *g.incident_edges(u).end(), [&](EdgeID e) {
        if (!visited[g.edge_target(e)]) {
          weights.push_back(g.edge_weight(e));
          validEdges.push_back(e);
        }
      });
      return DistributionDecorator<EdgeID, IndexDistributionWithoutReplacement>(
          weights.begin(), weights.end(), validEdges.begin(), validEdges.end()
      );
    };

    EdgeID localEdgesBurnt = 0;

    while (!activeNodes.empty()) {
      NodeID u = activeNodes.front();
      activeNodes.pop();

      tbb::concurrent_vector<EdgeID> validEdges;
      tbb::concurrent_vector<EdgeWeight> weights;
      tbb::parallel_for(*g.incident_edges(u).begin(), *g.incident_edges(u).end(), [&](EdgeID e) {
        if (!visited[g.edge_target(e)]) {
          weights.push_back(g.edge_weight(e));
          validEdges.push_back(e);
        }
      });
      DistributionDecorator<EdgeID, IndexDistributionWithoutReplacement> validNeighborDistribution(
          weights.begin(), weights.end(), validEdges.begin(), validEdges.end()
      );

      while (Random::instance().random_bool(_pf) ||
             validNeighborDistribution.underlying_distribution().empty()) {

        { // mark NodeID as visited, burn edge
          EdgeID e = validNeighborDistribution();
          NodeID x = g.edge_target(e);
          activeNodes.push(x);
          burnt[e]++;
          localEdgesBurnt++;
          visited[x] = true;
        }
      }
    }

    edges_burnt += localEdgesBurnt;
  }

  StaticArray<EdgeID> scores(g.m(), 0.0);
  tbb::parallel_for(0ul, scores.size(), [&](auto i) { scores[i] = burnt[i].load(); });

  // Not normalized as in the NetworKit implementation
  return scores;
}

} // namespace kaminpar::shm::sparsification
