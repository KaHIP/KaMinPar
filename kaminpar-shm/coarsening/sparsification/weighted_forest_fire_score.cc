#include "kaminpar-shm/coarsening/sparsification/weighted_forest_fire_score.h"

#include <queue>
#include <unordered_set>

#include "kaminpar-shm/coarsening/sparsification/sparsification_utils.h"

#include "kaminpar-common/datastructures/binary_heap.h"
#include "kaminpar-common/datastructures/fast_reset_array.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::sparsification {
StaticArray<EdgeID> WeightedForestFireScore::scores(const CSRGraph &g) {
  StaticArray<EdgeID> burnt(g.m());

  std::atomic<EdgeID> edges_burnt = 0;
  std::atomic<int> number_of_fires = 0;

  tbb::parallel_for(0, tbb::this_task_arena::max_concurrency(), [&](auto) {
    std::queue<NodeID> active_nodes;
    FastResetArray<bool> visited(g.n());
    Random &rand = Random::instance();
    DynamicBinaryMaxHeap<EdgeID, double> sample;

    std::vector<std::pair<EdgeID, double>> valid_edges_with_keys;

    constexpr EdgeID kNthElementCutoff = 1 << 10;
    constexpr std::size_t kPeriod = 1 << 14;

    std::vector<double> precomputed_log_doubles(kPeriod);
    for (double &value : precomputed_log_doubles) {
      value = std::log(rand.random_double());
    }

    std::size_t next_log_double = 0;
    auto get_next_log_double = [&] {
      next_log_double = (next_log_double + 1) % kPeriod;
      return precomputed_log_doubles[next_log_double];
    };
    const double log_pf = std::log(_pf);

    while (edges_burnt < _targetBurnRatio * g.m()) {
      number_of_fires++;

      // Start a new fire
      active_nodes.push(rand.random_index(0, g.n()));
      visited.clear();
      visited[active_nodes.front()] = true;

      EdgeID local_edges_burnt = 0;

      while (!active_nodes.empty()) {
        const NodeID u = active_nodes.front();
        active_nodes.pop();
        sample.clear();

        // Shifted geometric distribution
        const EdgeID neighbours_to_sample =
            static_cast<EdgeID>(std::ceil(get_next_log_double() / log_pf) - 1);
        if (neighbours_to_sample == 0) {
          continue;
        }

        if (neighbours_to_sample < kNthElementCutoff) {
          // We sample neighbors without replacement and probailies propotional to the weight of the
          // connecting edge using a version of the exponential clock method: Every incident edge
          // leading to an unvisited vertex is assined a random key depending on its weight and the
          // ones with the smallest keys are sampled.
          g.neighbors(u, [&](const EdgeID e, const NodeID v) {
            if (!visited.get(v)) {
              // Key for exponetial clock method
              const double key = -get_next_log_double() / (_ignore_weights ? 1 : g.edge_weight(e));

              if (sample.size() < neighbours_to_sample) {
                sample.push(e, key);
              } else if (key < sample.peek_key()) {
                sample.pop();
                sample.push(e, key);
              }
            }
          });

          for (const auto &entry : sample.elements()) {
            // Mark node as visited, burn edge
            const EdgeID e = entry.id;
            const NodeID v = g.edge_target(e);

            active_nodes.push(v);
            __atomic_add_fetch(&burnt[e], 1, __ATOMIC_RELAXED);
            local_edges_burnt++;
            visited[v] = true;
          }
        } else {
          valid_edges_with_keys.clear();

          // We sample neighbors without replacement and probailies propotional to the weight of the
          // connecting edge using a version of the exponential clock method: Every incident edge
          // leading to an unvisited vertex is assined a random key depending on its weight and the
          // ones with the smallest keys are sampled.
          g.neighbors(u, [&](const EdgeID e, const NodeID v) {
            if (!visited.get(v)) {
              // Key for exponetial clock method
              valid_edges_with_keys.emplace_back(
                  e, -get_next_log_double() / (_ignore_weights ? 1 : g.edge_weight(e))
              );
            }
          });

          if (valid_edges_with_keys.empty()) {
            continue;
          }

          const auto end_of_samping_range =
              valid_edges_with_keys.begin() +
              std::min<EdgeID>(neighbours_to_sample, valid_edges_with_keys.size());

          std::nth_element(
              valid_edges_with_keys.begin(),
              end_of_samping_range - 1,
              valid_edges_with_keys.end(),
              [](auto a, auto b) { return std::get<1>(a) < std::get<1>(b); }
          );

          for (auto p = valid_edges_with_keys.begin(); p != end_of_samping_range; ++p) {
            // Mark node as visited, burn edge
            const EdgeID e = p->first;
            const NodeID v = g.edge_target(e);

            active_nodes.push(v);
            __atomic_add_fetch(&burnt[e], 1, __ATOMIC_RELAXED);
            local_edges_burnt++;
            visited[v] = true;
          }
        }
      }

      edges_burnt += local_edges_burnt;
    }
  });

  print_fire_statistics(g, edges_burnt, number_of_fires);
  make_scores_symmetric(g, burnt);

  // Not normalized unlike in the NetworKit implementation
  return burnt;
}

void WeightedForestFireScore::make_scores_symmetric(
    const CSRGraph &g, StaticArray<EdgeID> &scores
) {
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
        EdgeID e1 = buckets[bucket_index][i].edge_id;
        EdgeID e2 = buckets[bucket_index][i + 1].edge_id;
        EdgeID combined_score = scores[e1] + scores[e2];
        scores[e1] = combined_score;
        scores[e2] = combined_score;
      }
    } else { // use hashed data structue
      std::unordered_set<EdgeWithEndpoints, EdgeWithEndpointHasher, EdgeWithEnpointComparator> set;
      for (EdgeID i = 0; i <= buckets[bucket_index].size(); i++) {
        EdgeWithEndpoints edge{};
        auto possible_counter_edge = set.extract(edge);

        if (possible_counter_edge.empty()) {
          set.insert(edge);
        } else {
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
