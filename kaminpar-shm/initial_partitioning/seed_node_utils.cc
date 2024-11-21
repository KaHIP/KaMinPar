/*******************************************************************************
 * Utility functions to find far-away nodes for BFS initialization.
 *
 * @file:   seed_node_utils.cc
 * @author: Daniel Seemaier
 * @date:   21.09.21
 ******************************************************************************/
#include "kaminpar-shm/initial_partitioning/seed_node_utils.h"

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/datastructures/queue.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm {

namespace {

std::pair<NodeID, NodeID> find_furthest_away_node(
    const CSRGraph &graph, const NodeID start_node, Queue<NodeID> &queue, Marker<> &marker
) {
  queue.push_tail(start_node);
  marker.set<true>(start_node);

  NodeID current_distance = 0;
  NodeID last_node = start_node;
  NodeID remaining_nodes_in_level = 1;
  NodeID nodes_in_next_level = 0;

  while (!queue.empty()) {
    const NodeID u = queue.head();
    queue.pop_head();
    last_node = u;

    graph.adjacent_nodes(u, [&](const NodeID v) {
      if (marker.get(v)) {
        return;
      }

      queue.push_tail(v);
      marker.set<true>(v);
      ++nodes_in_next_level;
    });

    // keep track of distance from start_node
    KASSERT(remaining_nodes_in_level > 0u);
    --remaining_nodes_in_level;
    if (remaining_nodes_in_level == 0) {
      ++current_distance;
      remaining_nodes_in_level = nodes_in_next_level;
      nodes_in_next_level = 0;
    }
  }
  KASSERT(current_distance > 0u);
  --current_distance;

  // bfs did not scan the whole graph, i.e., we have disconnected components
  if (marker.first_unmarked_element() < graph.n()) {
    last_node = marker.first_unmarked_element();
    current_distance = std::numeric_limits<NodeID>::max(); // infinity
  }

  marker.reset();
  queue.clear();
  return {last_node, current_distance};
}

} // namespace

std::pair<NodeID, NodeID> find_far_away_nodes(const CSRGraph &graph, const int num_iterations) {
  Queue<NodeID> queue(graph.n());
  Marker<> marker(graph.n());

  return find_far_away_nodes(graph, num_iterations, queue, marker);
}

std::pair<NodeID, NodeID> find_far_away_nodes(
    const CSRGraph &graph, int num_iterations, Queue<NodeID> &queue, Marker<> &marker
) {
  queue.clear();
  marker.reset();

  NodeID best_distance = 0;
  std::pair<NodeID, NodeID> best_pair = {0, 0};

  Random &rand = Random::instance();

  for (int i = 0; i < num_iterations; ++i) {
    const NodeID u = Random::instance().random_index(0, graph.n());
    const auto [v, distance] = find_furthest_away_node(graph, u, queue, marker);

    if (distance > best_distance || (distance == best_distance && rand.random_bool())) {
      best_distance = distance;
      best_pair = {u, v};
    }
  }

  return best_pair;
}

} // namespace kaminpar::shm
