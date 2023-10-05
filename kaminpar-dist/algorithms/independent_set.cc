/*******************************************************************************
 * Basic independent set algorithm for distributed graphs.
 *
 * @file:   independent_set.cc
 * @author: Daniel Seemaier
 * @date:   22.08.2022
 ******************************************************************************/
#include "kaminpar-dist/algorithms/independent_set.h"

#include <algorithm>
#include <limits>
#include <random>

#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"

#include "kaminpar-common/timer.h"

namespace kaminpar::dist::graph {
SET_DEBUG(false);

namespace {
template <typename Generator, typename ScoreType = std::int64_t>
ScoreType compute_score(Generator &generator, const GlobalNodeID node, const int seed) {
  // @todo replace with something efficient / use communication
  generator.seed(seed + node);
  return std::uniform_int_distribution<ScoreType>(
      ScoreType(), std::numeric_limits<ScoreType>::max() - 1
  )(generator);
}
} // namespace

std::vector<NodeID>
find_independent_border_set(const DistributedPartitionedGraph &p_graph, const int seed) {
  constexpr std::int64_t kNoBorderNode = std::numeric_limits<std::int64_t>::max();

  NoinitVector<std::int64_t> score(p_graph.total_n());
  tbb::enumerable_thread_specific<std::mt19937> generator_ets;

  p_graph.pfor_all_nodes([&](const NodeID u) {
    if (!p_graph.is_owned_node(u)) {
      // Compute score for ghost nodes lazy
      score[u] = -1;
    } else if (p_graph.check_border_node(u)) {
      // Compute score for owned border nodes
      score[u] = compute_score(generator_ets.local(), p_graph.local_to_global_node(u), seed);
    } else {
      // Otherwise mark node as non-border node
      score[u] = kNoBorderNode;
    }
  });

  // Select nodes that have the lowest score in their neighborhood as
  // independent set
  tbb::concurrent_vector<NodeID> seed_nodes;

  p_graph.pfor_nodes([&](const NodeID u) {
    if (score[u] == kNoBorderNode) {
      return; // Not a border node
    }

    const bool is_seed_node = std::all_of(
        p_graph.adjacent_nodes(u).begin(),
        p_graph.adjacent_nodes(u).end(),
        [&](const NodeID v) {
          // Compute score for ghost nodes lazy
          if (score[v] < 0) {
            const auto v_score =
                compute_score(generator_ets.local(), p_graph.local_to_global_node(v), seed);
            __atomic_store_n(&score[v], v_score, __ATOMIC_RELAXED);
          }

          return score[u] < score[v];
        }
    );

    if (is_seed_node) {
      seed_nodes.push_back(u);
    }
  });

  return std::vector<NodeID>(seed_nodes.begin(), seed_nodes.end());
}
} // namespace kaminpar::dist::graph
