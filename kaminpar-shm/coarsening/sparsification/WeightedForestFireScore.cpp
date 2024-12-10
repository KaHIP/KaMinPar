#include "WeightedForestFireScore.h"

#include <functional>
#include <queue>

#include <oneapi/tbb/concurrent_vector.h>

#include "DistributionDecorator.h"
#include "sparsification_utils.h"

#include "kaminpar-common/random.h"

namespace kaminpar::shm::sparsification {
StaticArray<EdgeID> WeightedForestFireScore::scores(const CSRGraph &g) {
  StaticArray<EdgeID> burnt(g.m());
  tbb::parallel_for(0ul, burnt.size(), [&](auto i) { burnt[i] = 0; });
  EdgeID edges_burnt = 0;

  tbb::concurrent_vector<EdgeID> numbers_of_edges_burnt;

  int number_of_fires = 0;
  tbb::parallel_for(0, tbb::this_task_arena::max_concurrency(), [&](auto) {
    while (edges_burnt < _targetBurnRatio * g.m()) {
      __atomic_fetch_add(&number_of_fires, 1, __ATOMIC_RELAXED);
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
        for (EdgeID e : g.incident_edges(u)) {
          if (!visited[g.edge_target(e)]) {
            weights.push_back(g.edge_weight(e));
            validEdges.push_back(e);
          }
        }

        EdgeID neighbours_to_sample = std::min(
            static_cast<EdgeID>(
                std::ceil(std::log(Random::instance().random_double()) / std::log(_pf)) - 1
            ), // shifted geometric distribution
            static_cast<EdgeID>(validEdges.size())
        );
        auto sampled_neighbours_indices = utils::sample_k_without_replacement(
            weights.begin(), weights.end(), neighbours_to_sample
        );
        for (auto i : sampled_neighbours_indices) {
          // mark NodeID as visited, burn edge
          EdgeID e = validEdges[i];
          NodeID x = g.edge_target(e);
          activeNodes.push(x);
          __atomic_add_fetch(&burnt[e], 1, __ATOMIC_RELAXED);
          localEdgesBurnt++;
          visited[x] = true;
        }
      }
      __atomic_add_fetch(&edges_burnt, localEdgesBurnt, __ATOMIC_RELAXED);
      numbers_of_edges_burnt.push_back(localEdgesBurnt);
    }
  });

  print_fire_statistics(g, edges_burnt, number_of_fires, numbers_of_edges_burnt);

  // Not normalized unlike in the NetworKit implementation
  return burnt;
}

void WeightedForestFireScore::print_fire_statistics(
    const CSRGraph &g,
    EdgeID edges_burnt,
    int number_of_fires,
    tbb::concurrent_vector<EdgeID> numbers_of_edges_burnt
) {
  const auto default_precision{std::cout.precision()};
  std::cout << std::setprecision(4);

  double average = static_cast<double>(edges_burnt) / number_of_fires;
  double variance = 0;
  for (auto x : numbers_of_edges_burnt) {
    double local_burnt = static_cast<double>(x);
    variance += (local_burnt - average) * (local_burnt - average) / (number_of_fires - 1);
  }

  std::cout << "** targetBurntRatio=" << _targetBurnRatio << ", pf=" << _pf << "\n";
  std::cout << "** m=" << g.m() << ", n=" << g.n() << "\n";
  std::cout << "** " << number_of_fires << " fires have burned " << edges_burnt << " edges\n";
  std::cout << "** edges burnt per fire: avg=" << average << ", var=" << variance << " min="
            << *std::min_element(numbers_of_edges_burnt.begin(), numbers_of_edges_burnt.end())
            << ", max="
            << *std::max_element(numbers_of_edges_burnt.begin(), numbers_of_edges_burnt.end())
            << "\n";

  std::cout << std::setprecision(default_precision);
}

} // namespace kaminpar::shm::sparsification
