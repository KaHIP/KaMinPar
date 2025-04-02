#include "WeightedForestFireScore.h"

#include <functional>
#include <queue>
#include <set>

#include "DistributionDecorator.h"
#include "sparsification_utils.h"

#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

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
  SCOPED_TIMER("Make Scores Symetric");
  // TODO tune these constants
  const EdgeID AVERAGE_EDGES_PER_BUCKET = 1000;
  const EdgeID HASHING_THRESHOLD = 20 * AVERAGE_EDGES_PER_BUCKET;

  const EdgeID number_of_buckets =
      std::max(g.m() / AVERAGE_EDGES_PER_BUCKET, static_cast<EdgeID>(1));

  std::vector<std::vector<EdgeWithEndpoints>> buckets(number_of_buckets);
  std::vector<tbb::spin_mutex> bucket_locks(number_of_buckets);

  sparsification::utils::parallel_for_edges_with_endpoints(g, [&](EdgeID e, NodeID u, NodeID v) {
    EdgeWithEndpoints edge_with_endpoints =
        u < v ? EdgeWithEndpoints{e, u, v} : EdgeWithEndpoints{e, v, u};
    EdgeID bucket_index = hash(edge_with_endpoints) % number_of_buckets;

    bucket_locks[bucket_index].lock();
    buckets[bucket_index].push_back(edge_with_endpoints);
    bucket_locks[bucket_index].unlock();
  });

  tbb::parallel_for<EdgeID>(0, number_of_buckets, [&](auto bucket_index) {
    KASSERT(buckets[bucket_index].size() % 2 == 0, "odd size of bucket", assert::always);
    if (buckets[bucket_index].size() < HASHING_THRESHOLD) {
      std::sort(
          buckets[bucket_index].begin(),
          buckets[bucket_index].end(),
          [](const EdgeWithEndpoints &e1, const EdgeWithEndpoints &e2) {
            if (e1.smaller_endpoint != e2.smaller_endpoint)
              return e1.smaller_endpoint < e2.smaller_endpoint;
            return e1.larger_endpoint < e2.larger_endpoint;
          }
      );

      for (EdgeID i = 0; i < buckets[bucket_index].size(); i += 2) {
        KASSERT(
            buckets[bucket_index][i].smaller_endpoint ==
                    buckets[bucket_index][i + 1].smaller_endpoint &&
                buckets[bucket_index][i].larger_endpoint ==
                    buckets[bucket_index][i + 1].larger_endpoint,
            "the wrong edges were matched",
            assert::always
        );

        EdgeID e1 = buckets[bucket_index][i].edge_id;
        EdgeID e2 = buckets[bucket_index][i + 1].edge_id;
        EdgeID combined_score = scores[e1] + scores[e2];
        scores[e1] = combined_score;
        scores[e2] = combined_score;
      }
    } else { // use hashed data structue
      std::unordered_set<EdgeWithEndpoints, EdgeWithEndpointHasher, EdgeWithEnpointComparator> set;
      for (EdgeID i = 0; i <= buckets[bucket_index].size(); i++) {
        EdgeWithEndpoints edge;
        auto possible_counter_edge = set.extract(edge);

        if (possible_counter_edge.empty()) {
          set.insert(edge);
        } else {
          KASSERT(EdgeWithEnpointComparator()(edge, possible_counter_edge.value()), "counter edge does not match", assert::always);
          EdgeID counter_edge_id = possible_counter_edge.value().edge_id;
          EdgeID combined_score = scores[edge.edge_id] + scores[counter_edge_id];
          scores[edge.edge_id] = combined_score;
          scores[counter_edge_id] = combined_score;
        }
      }
    }
  });
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
