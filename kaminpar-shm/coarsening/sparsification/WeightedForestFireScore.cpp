#include "WeightedForestFireScore.h"

#include <functional>
#include <queue>
#include <set>

#include "DistributionDecorator.h"
#include "sparsification_utils.h"

#include "kaminpar-common/random.h"

namespace kaminpar::shm::sparsification {
StaticArray<EdgeID> WeightedForestFireScore::scores(const CSRGraph &g) {
  StaticArray<EdgeID> burnt(g.m(), 0);
  EdgeID edges_burnt = 0;

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
      visited.insert(activeNodes.front());

      EdgeID localEdgesBurnt = 0;

      while (!activeNodes.empty()) {
        const NodeID u = activeNodes.front();
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
        const EdgeID neighbours_to_sample = std::min(
            static_cast<EdgeID>(
                std::ceil(std::log(Random::instance().random_double()) / std::log(_pf)) - 1
            ), // shifted geometric distribution
            static_cast<EdgeID>(validEdgesWithKeys.size())
        );
        KASSERT(neighbours_to_sample <= validEdgesWithKeys.size());
        if (neighbours_to_sample == 0)
          continue;

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
    }
  });

  print_fire_statistics(g, edges_burnt, number_of_fires);

  make_scores_symetric(g, burnt);

  // Not normalized unlike in the NetworKit implementation
  return burnt;
}

void WeightedForestFireScore::make_scores_symetric(const CSRGraph &g, StaticArray<EdgeID> &scores) {
  const EdgeID average_edes_per_bucket = 100; // TODO what's a suitable constant?

  // TODO use static arrays after figuring out how to initalize them when they are nested
  std::vector<std::vector<std::vector<EdgeID>>> buckets(g.n());
  std::vector<std::vector<EdgeID>> edges_done(g.n());
  StaticArray<EdgeID> number_of_buckets(g.n());

  // TODO use actual hash fuction
  auto hash = [&](NodeID source, NodeID target) {
    return target % number_of_buckets[source];
  };

  //g.pfor_nodes([&](NodeID v) {
  for (NodeID v =0; v < g.n(); v++){
    number_of_buckets[v] = (g.degree(v) + average_edes_per_bucket - 1) / average_edes_per_bucket;

    buckets[v] = std::vector<std::vector<EdgeID>>(number_of_buckets[v]);

    for (EdgeID e : g.incident_edges(v)) {
      EdgeID bucket_index = hash(v, g.edge_target(e));
      KASSERT(bucket_index < number_of_buckets[v], "", assert::always);
      KASSERT(number_of_buckets[v] == buckets[v].size(), "", assert::always);
      buckets[v][bucket_index].push_back(e);
    }

    for (size_t i = 0; i < number_of_buckets[v]; i++) {
      std::sort(buckets[v][i].begin(), buckets[v][i].end(), [&](EdgeID e1, EdgeID e2) {
        return g.edge_target(e1) <= g.edge_target(e2);
      });
    }

    edges_done[v] = std::vector<EdgeID>(number_of_buckets[v],0);
  }//);

  for (NodeID u : g.nodes()) { // TODO can this be parallel without breaking the algo
    for (EdgeID e : g.incident_edges(u)) {
      NodeID v = g.edge_target(e);
      EdgeID bucket_index = hash(v, u);
      EdgeID counter_edge = buckets[v][bucket_index][edges_done[v][bucket_index]++];
      KASSERT(u == g.edge_target(counter_edge), "not the real counter edge", assert::always);

      EdgeID combined_scores = scores[e] + scores[counter_edge];
      scores[e] = combined_scores;
      scores[counter_edge] = combined_scores;
    }
  }
}

void WeightedForestFireScore::print_fire_statistics(
    const CSRGraph &g, EdgeID edges_burnt, int number_of_fires
) {
  const auto default_precision{std::cout.precision()};
  std::cout << std::setprecision(4);

  std::cout << "** targetBurntRatio=" << _targetBurnRatio << ", pf=" << _pf << "\n";
  std::cout << "** m=" << g.m() << ", n=" << g.n() << "\n";
  std::cout << "** " << number_of_fires << " fires have burned " << edges_burnt << " edges\n";

  std::cout << std::setprecision(default_precision);
}

} // namespace kaminpar::shm::sparsification
