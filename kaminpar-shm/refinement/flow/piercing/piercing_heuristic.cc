#include "kaminpar-shm/refinement/flow/piercing/piercing_heuristic.h"

#include <queue>
#include <utility>

namespace kaminpar::shm {

PiercingHeuristic::PiercingHeuristic(
    const CSRGraph &graph,
    const std::unordered_set<NodeID> &initial_source_side_nodes,
    const std::unordered_set<NodeID> &initial_sink_side_nodes
)
    : _graph(graph),
      _initial_source_side_nodes(initial_source_side_nodes),
      _initial_sink_side_nodes(initial_sink_side_nodes) {
  compute_distances();
}

NodeID PiercingHeuristic::pierce_on_source_side(
    const std::unordered_set<NodeID> &source_side_cut,
    const std::unordered_set<NodeID> &sink_side_cut,
    const std::unordered_set<NodeID> &sink_side_nodes,
    NodeWeight max_piercing_node_weight
) {
  return find_piercing_node(
      source_side_cut,
      sink_side_cut,
      sink_side_nodes,
      _initial_source_side_nodes,
      max_piercing_node_weight
  );
}

NodeID PiercingHeuristic::pierce_on_sink_side(
    const std::unordered_set<NodeID> &sink_side_cut,
    const std::unordered_set<NodeID> &source_side_cut,
    const std::unordered_set<NodeID> &source_side_nodes,
    NodeWeight max_piercing_node_weight
) {
  return find_piercing_node(
      sink_side_cut,
      source_side_cut,
      source_side_nodes,
      _initial_sink_side_nodes,
      max_piercing_node_weight
  );
}

NodeID PiercingHeuristic::find_piercing_node(
    const std::unordered_set<NodeID> &terminal_cut,
    const std::unordered_set<NodeID> &other_terminal_cut,
    const std::unordered_set<NodeID> &other_terminal_side_nodes,
    const std::unordered_set<NodeID> &initial_terminal_side_nodes,
    const NodeWeight max_piercing_node_weight
) {
  NodeID piercing_node = kInvalidNodeID;

  NodeWeight cur_weight = 0;
  NodeID cur_distance = 0;
  bool avoided_augmenting_path = false;
  for (const NodeID u : terminal_cut) {
    _graph.adjacent_nodes(u, [&](const NodeID v) {
      if (terminal_cut.contains(v) || other_terminal_side_nodes.contains(v)) {
        return;
      }

      const NodeWeight v_weight = _graph.node_weight(v);
      if (v_weight > max_piercing_node_weight) {
        return;
      }

      const bool avoids_augmenting_path = !other_terminal_cut.contains(v);
      const NodeID distance = initial_terminal_side_nodes.contains(v) ? _distance[v] : 0;

      if (piercing_node == kInvalidNodeID) {
        piercing_node = v;
        cur_weight = v_weight;
        cur_distance = distance;
        avoided_augmenting_path = avoids_augmenting_path;
        return;
      }

      if (avoided_augmenting_path) {
        if (avoids_augmenting_path && v_weight > cur_weight) {
          piercing_node = v;
          cur_weight = v_weight;
          cur_distance = distance;
        }
      } else {
        if (avoids_augmenting_path) {
          piercing_node = v;
          cur_weight = v_weight;
          cur_distance = distance;
          avoided_augmenting_path = true;
        } else if (cur_distance < distance) {
          piercing_node = v;
          cur_weight = v_weight;
          cur_distance = distance;
        }
      }
    });
  }

  return piercing_node;
}

void PiercingHeuristic::compute_distances() {
  if (_distance.size() < _graph.n()) {
    _distance.resize(_graph.n(), kInvalidNodeID, static_array::seq);
  }

  std::queue<std::pair<NodeID, NodeID>> source_bfs_queue;
  std::queue<std::pair<NodeID, NodeID>> sink_bfs_queue;
  for (const NodeID u : _graph.nodes()) {
    if (!_initial_source_side_nodes.contains(u)) {
      continue;
    }

    bool has_cut_edge = false;
    _graph.adjacent_nodes(u, [&](const NodeID v) {
      if (!_initial_sink_side_nodes.contains(v)) {
        return;
      }

      has_cut_edge = true;

      if (_distance[v] == kInvalidNodeID) {
        _distance[v] = 1;
        sink_bfs_queue.emplace(v, 1);
      }
    });

    if (has_cut_edge) {
      _distance[u] = 1;
      source_bfs_queue.emplace(u, 1);
    }
  }

  const auto perform_bfs = [&](auto &bfs_queue, const auto &nodes) {
    while (!bfs_queue.empty()) {
      const auto [u, u_distance] = bfs_queue.front();
      bfs_queue.pop();

      const NodeID v_distance = u_distance + 1;
      _graph.adjacent_nodes(u, [&](const NodeID v) {
        if (_distance[v] != kInvalidNodeID || !nodes.contains(v)) {
          return;
        }

        _distance[v] = v_distance;
        bfs_queue.emplace(v, v_distance);
      });
    }
  };

  perform_bfs(source_bfs_queue, _initial_source_side_nodes);
  perform_bfs(sink_bfs_queue, _initial_sink_side_nodes);
}

} // namespace kaminpar::shm
