#include "WeightedForestFireScore.h"

#include <functional>
#include <queue>

#include <oneapi/tbb/concurrent_vector.h>

#include "DistributionDecorator.h"
#include "IndexDistributionWithoutReplacement.h"

#include "kaminpar-common/random.h"

namespace kaminpar::shm::sparsification {
StaticArray<EdgeID> WeightedForestFireScore::scores(const CSRGraph &g) {
  StaticArray<EdgeID> burnt(g.m());
  tbb::parallel_for(0ul, burnt.size(), [&](auto i) { burnt[i] = 0; });
  EdgeID edges_burnt = 0;

  tbb::parallel_for(0, tbb::this_task_arena::max_concurrency(), [&](auto) {
    while (edges_burnt < _targetBurnRatio * g.m()) {
      // Start a new fire
      std::queue<NodeID> activeNodes;
      StaticArray<bool> visited(g.m(), false);
      activeNodes.push(Random::instance().random_index(0, g.n()));

      EdgeID localEdgesBurnt = 0;

      while (!activeNodes.empty()) {
        NodeID u = activeNodes.front();
        activeNodes.pop();

        std::vector<EdgeID> validEdges;
        std::vector<EdgeWeight> weights;
        for (EdgeID e: g.incident_edges(u)) {
          if (!visited[g.edge_target(e)]) {
            weights.push_back(g.edge_weight(e));
            validEdges.push_back(e);
          }
        }
        DistributionDecorator<EdgeID, IndexDistributionWithoutReplacement>
            validNeighborDistribution(
                weights.begin(), weights.end(), validEdges.begin(), validEdges.end()
            );

        while (Random::instance().random_bool(_pf) ||
               validNeighborDistribution.underlying_distribution().empty()) {

          { // mark NodeID as visited, burn edge
            EdgeID e = validNeighborDistribution();
            NodeID x = g.edge_target(e);
            activeNodes.push(x);
            __atomic_add_fetch(&burnt[e], 1, __ATOMIC_RELAXED);
            localEdgesBurnt++;
            visited[x] = true;
          }
        }
      }

      __atomic_add_fetch(&edges_burnt, localEdgesBurnt, __ATOMIC_RELAXED);
    }
  });

  // Not normalized as in the NetworKit implementation
  return burnt;
}

} // namespace kaminpar::shm::sparsification
