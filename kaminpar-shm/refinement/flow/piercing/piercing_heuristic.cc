#include "kaminpar-shm/refinement/flow/piercing/piercing_heuristic.h"

#include <algorithm>
#include <queue>
#include <utility>

namespace kaminpar::shm {

PiercingHeuristic::PiercingHeuristic(const PiercingHeuristicContext &ctx) : _ctx(ctx) {}

void PiercingHeuristic::initialize(
    const CSRGraph &graph,
    const std::unordered_set<NodeID> &initial_source_side_nodes,
    const std::unordered_set<NodeID> &initial_sink_side_nodes
) {
  _graph = &graph;
  _initial_source_side_nodes = &initial_source_side_nodes;
  _initial_sink_side_nodes = &initial_sink_side_nodes;

  compute_distances();
}

std::span<const NodeID> PiercingHeuristic::pierce_on_source_side(
    std::span<const NodeID> piercing_nodes_candidates,
    const NodeStatus &cut_status,
    const NodeStatus &terminal_status,
    const NodeWeight max_weight
) {
  return find_piercing_node(
      piercing_nodes_candidates,
      cut_status,
      terminal_status,
      *_initial_source_side_nodes,
      max_weight,
      true
  );
}

std::span<const NodeID> PiercingHeuristic::pierce_on_sink_side(
    std::span<const NodeID> piercing_nodes_candidates,
    const NodeStatus &cut_status,
    const NodeStatus &terminal_status,
    const NodeWeight max_weight
) {
  return find_piercing_node(
      piercing_nodes_candidates,
      cut_status,
      terminal_status,
      *_initial_sink_side_nodes,
      max_weight,
      false
  );
}

std::span<const NodeID> PiercingHeuristic::find_piercing_node(
    std::span<const NodeID> piercing_nodes_candidates,
    const NodeStatus &cut_status,
    const NodeStatus &terminal_status,
    const std::unordered_set<NodeID> &initial_terminal_side_nodes,
    const NodeWeight max_weight,
    const bool source_side
) {
  const std::uint8_t side_status = source_side ? NodeStatus::kSource : NodeStatus::kSink;
  const std::uint8_t other_side_status = source_side ? NodeStatus::kSink : NodeStatus::kSource;

  NodeWeight cur_weight = 0;
  NodeID cur_distance = 0;
  bool avoided_augmenting_path = false;

  _piercing_nodes.clear();
  for (const NodeID u : piercing_nodes_candidates) {
    const NodeWeight u_weight = _graph->node_weight(u);
    const bool avoids_augmenting_path = !cut_status.has_status(u, other_side_status);
    const NodeID distance = initial_terminal_side_nodes.contains(u) ? _distance[u] : 0;

    if (_piercing_nodes.empty()) {
      if (u_weight <= max_weight) {
        _piercing_nodes.push_back(u);
        cur_weight = u_weight;

        cur_distance = distance;
        avoided_augmenting_path = avoids_augmenting_path;
      }

      continue;
    }

    if (avoided_augmenting_path) {
      if (avoids_augmenting_path && cur_weight + u_weight <= max_weight) {
        _piercing_nodes.push_back(u);
        cur_weight += u_weight;
      }

      continue;
    }

    if (u_weight > max_weight) {
      continue;
    }

    if (avoids_augmenting_path) {
      _piercing_nodes[0] = u;
      cur_weight = u_weight;

      if (_ctx.pierce_all_viable) {
        avoided_augmenting_path = true;
      } else {
        break;
      }
    } else if (cur_distance < distance) {
      _piercing_nodes[0] = u;
      cur_weight = u_weight;

      cur_distance = distance;
    }
  }

  if (_piercing_nodes.empty()) {
    NodeID cur_piercing_node = kInvalidNodeID;

    for (const NodeID u : _graph->nodes()) {
      if (cut_status.has_status(u, side_status) ||
          terminal_status.has_status(u, other_side_status)) {
        continue;
      }

      const NodeWeight u_weight = _graph->node_weight(u);
      if (u_weight > max_weight) {
        continue;
      }

      const NodeID distance = initial_terminal_side_nodes.contains(u) ? _distance[u] : 0;
      if (cur_piercing_node == kInvalidNodeID || cur_distance < distance) {
        cur_piercing_node = u;
        cur_distance = distance;
      }
    }

    if (cur_piercing_node != kInvalidNodeID) {
      _piercing_nodes.push_back(cur_piercing_node);
    }
  }

  return _piercing_nodes;
}

void PiercingHeuristic::compute_distances() {
  if (_distance.size() < _graph->n()) {
    _distance.resize(_graph->n(), static_array::noinit);
  }
  std::fill_n(_distance.begin(), _graph->n(), kInvalidNodeID);

  std::queue<std::pair<NodeID, NodeID>> source_bfs_queue;
  std::queue<std::pair<NodeID, NodeID>> sink_bfs_queue;
  for (const NodeID u : _graph->nodes()) {
    if (!_initial_source_side_nodes->contains(u)) {
      continue;
    }

    bool has_cut_edge = false;
    _graph->adjacent_nodes(u, [&](const NodeID v) {
      if (!_initial_sink_side_nodes->contains(v)) {
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
      _graph->adjacent_nodes(u, [&](const NodeID v) {
        if (_distance[v] != kInvalidNodeID || !nodes.contains(v)) {
          return;
        }

        _distance[v] = v_distance;
        bfs_queue.emplace(v, v_distance);
      });
    }
  };

  perform_bfs(source_bfs_queue, *_initial_source_side_nodes);
  perform_bfs(sink_bfs_queue, *_initial_sink_side_nodes);
}

} // namespace kaminpar::shm
