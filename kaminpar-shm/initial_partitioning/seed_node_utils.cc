/*******************************************************************************
 * @file:   seed_node_utils.cc
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Algorithms to find seed nodes for initial partitioner based on
 * graph growing.
 ******************************************************************************/
#include "kaminpar-shm/initial_partitioning/seed_node_utils.h"

#include "kaminpar-shm/definitions.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm::ip {
std::pair<NodeID, NodeID>
find_far_away_nodes(const Graph &graph, const std::size_t num_iterations) {
  Queue<NodeID> queue(graph.n());
  Marker<> marker(graph.n());

  NodeID best_distance = 0;
  std::pair<NodeID, NodeID> best_pair{0, 0};
  for (std::size_t i = 0; i < num_iterations; ++i) {
    const NodeID u = Random::instance().random_index(0, graph.n());
    const auto [v, distance] = find_furthest_away_node(graph, u, queue, marker);

    if (distance > best_distance ||
        (distance == best_distance && Random::instance().random_bool())) {
      best_distance = distance;
      best_pair = {u, v};
    }
  }

  return best_pair;
}

std::pair<NodeID, NodeID> find_furthest_away_node(
    const Graph &graph, const NodeID start_node, Queue<NodeID> &queue, Marker<> &marker
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

    for (const NodeID v : graph.adjacent_nodes(u)) {
      if (marker.get(v))
        continue;
      queue.push_tail(v);
      marker.set<true>(v);
      ++nodes_in_next_level;
    }

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
} // namespace kaminpar::shm::ip
