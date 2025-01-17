#include "WeightedForestFireScore.h"

#include <functional>
#include <queue>
#include <set>

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
    // Preallocate everything here
    std::queue<NodeID> activeNodes;
    std::unordered_set<EdgeID> visited;
    std::vector<std::pair<EdgeID, double>> validEdgesWithKeys;

    while (edges_burnt < _targetBurnRatio * g.m()) {
      __atomic_fetch_add(&number_of_fires, 1, __ATOMIC_RELAXED);
      // Start a new fire
      visited.clear();
      activeNodes.push(Random::instance().random_index(0, g.n()));

      EdgeID localEdgesBurnt = 0;

      while (!activeNodes.empty()) {
        NodeID u = activeNodes.front();
        activeNodes.pop();

        validEdgesWithKeys.clear();

        // We sample neighbors without replacement and probailies propotional to the weight of the
        // connecting edge using a version of the exponential clock method: Every incident edge
        // leading to an unvisited vertex is assined a random key depending on its weight and the
        // ones with the smallest keys are sampled.
        for (EdgeID e : g.incident_edges(u)) {
          if (!visited.contains(g.edge_target(e))) {
            validEdgesWithKeys.emplace_back(
                e,
                // key for exponetial clock method
                -std::log(Random::instance().random_double()) / g.edge_weight(e)
            );
          }
        }
        EdgeID neighbours_to_sample = std::min(
            static_cast<EdgeID>(
                std::ceil(std::log(Random::instance().random_double()) / std::log(_pf)) - 1
            ), // shifted geometric distribution
            static_cast<EdgeID>(validEdgesWithKeys.size())
        );
        auto end_of_samping_range = validEdgesWithKeys.begin() + neighbours_to_sample;
        std::nth_element(
            validEdgesWithKeys.begin(),
            end_of_samping_range - 1,
            validEdgesWithKeys.end(),
            [](auto a, auto b) {
              return std::get<1>(a) < std::get<1>(b);
            } // comp keys
        );

        for (auto p = validEdgesWithKeys.begin(); p != end_of_samping_range; ++p) {
          // mark NodeID as visited, burn edge
          auto [e, _] = *p;
          NodeID v = g.edge_target(e);
          activeNodes.push(v);
          __atomic_add_fetch(&burnt[e], 1, __ATOMIC_RELAXED);
          localEdgesBurnt++;
          visited.insert(v);
        }
      }
      __atomic_add_fetch(&edges_burnt, localEdgesBurnt, __ATOMIC_RELAXED);
      // numbers_of_edges_burnt.push_back(localEdgesBurnt);
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
  /*
  std::cout << "** edges burnt per fire: avg=" << average << ", var=" << variance << " min="
            << *std::min_element(numbers_of_edges_burnt.begin(), numbers_of_edges_burnt.end())
            << ", max="
            << *std::max_element(numbers_of_edges_burnt.begin(), numbers_of_edges_burnt.end())
            << "\n";
  */

  std::cout << std::setprecision(default_precision);
}

} // namespace kaminpar::shm::sparsification
