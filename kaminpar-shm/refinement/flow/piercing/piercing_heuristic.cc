#include "kaminpar-shm/refinement/flow/piercing/piercing_heuristic.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <utility>

namespace kaminpar::shm {

PiercingHeuristic::PiercingHeuristic(const PiercingHeuristicContext &ctx) : _ctx(ctx) {}

void PiercingHeuristic::initialize(
    const CSRGraph &graph,
    const std::unordered_set<NodeID> &initial_source_side_nodes,
    const std::unordered_set<NodeID> &initial_sink_side_nodes,
    const NodeWeight source_side_weight,
    const NodeWeight sink_side_weight,
    const NodeWeight total_weight,
    const NodeWeight max_source_side_weight,
    const NodeWeight max_sink_side_weight
) {
  _graph = &graph;
  _initial_source_side_nodes = &initial_source_side_nodes;
  _initial_sink_side_nodes = &initial_sink_side_nodes;

  const NodeID max_distance = compute_distances();
  _reachable_candidates_buckets.initialize(max_distance);
  _unreachable_candidates_buckets.initialize(max_distance);

  const NodeWeight max_total_weight = max_source_side_weight + max_sink_side_weight;
  _source_side_bulk_piercing_ctx.initialize(
      source_side_weight, total_weight, max_source_side_weight, max_total_weight
  );
  _sink_side_bulk_piercing_ctx.initialize(
      sink_side_weight, total_weight, max_sink_side_weight, max_total_weight
  );
}

void PiercingHeuristic::reset(const bool source_side) {
  _source_side = source_side;

  _initial_side_nodes = source_side ? _initial_source_side_nodes : _initial_sink_side_nodes;

  _reachable_candidates_buckets.reset();
  _unreachable_candidates_buckets.reset();
}

void PiercingHeuristic::add_piercing_node_candidate(const NodeID node, const bool reachable) {
  const NodeID distance = _initial_side_nodes->contains(node) ? _distance[node] : 0;

  if (reachable) {
    _reachable_candidates_buckets.add_candidate(node, distance);
  } else {
    _unreachable_candidates_buckets.add_candidate(node, distance);
  }
}

std::span<const NodeID> PiercingHeuristic::find_piercing_nodes(
    const NodeStatus &cut_status,
    const NodeStatus &terminal_status,
    const NodeWeight side_weight,
    const NodeWeight max_weight
) {
  _piercing_nodes.clear();

  NodeWeight cur_weight = 0;
  const auto add_piercing_nodes = [&](const auto &candidates_buckets,
                                      const auto max_num_piercing_nodes) {
    const std::int64_t max_bucket = _unreachable_candidates_buckets.max_occupied_bucket();
    const std::int64_t min_bucket = _unreachable_candidates_buckets.min_occupied_bucket();

    for (std::int64_t bucket = max_bucket; bucket >= min_bucket; --bucket) {
      for (const NodeID u : candidates_buckets.candidates(bucket)) {
        const NodeWeight u_weight = _graph->node_weight(u);
        if (cur_weight + u_weight > max_weight) {
          continue;
        }

        cur_weight += u_weight;
        _piercing_nodes.push_back(u);

        if (_piercing_nodes.size() >= max_num_piercing_nodes) {
          return;
        }
      }
    }
  };

  const std::size_t max_num_piercing_nodes = compute_max_num_piercing_nodes(side_weight);
  add_piercing_nodes(
      _unreachable_candidates_buckets,
      _ctx.pierce_all_viable ? std::numeric_limits<std::size_t>::max() : max_num_piercing_nodes
  );
  add_piercing_nodes(_reachable_candidates_buckets, max_num_piercing_nodes);

  if (_ctx.bulk_piercing) {
    BulkPiercingContext &bp_ctx = bulk_piercing_context();
    bp_ctx.total_bulk_piercing_nodes += _piercing_nodes.size();
  }

  if (_ctx.fallback_heuristic && _piercing_nodes.empty()) {
    const std::uint8_t side_status = _source_side ? NodeStatus::kSource : NodeStatus::kSink;
    const std::uint8_t other_side_status = _source_side ? NodeStatus::kSink : NodeStatus::kSource;
    const std::unordered_set<NodeID> &initial_terminal_side_nodes =
        _source_side ? *_initial_source_side_nodes : *_initial_sink_side_nodes;

    NodeID cur_piercing_node = kInvalidNodeID;
    NodeID cur_distance = kInvalidNodeID;
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

NodeID PiercingHeuristic::compute_distances() {
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

  NodeID max_distance = 1;
  const auto perform_bfs = [&](auto &bfs_queue, const auto &nodes) {
    while (!bfs_queue.empty()) {
      const auto [u, u_distance] = bfs_queue.front();
      bfs_queue.pop();

      const NodeID v_distance = u_distance + 1;
      _graph->adjacent_nodes(u, [&](const NodeID v) {
        if (_distance[v] != kInvalidNodeID || !nodes.contains(v)) {
          return;
        }

        max_distance = std::max(max_distance, v_distance);
        _distance[v] = v_distance;
        bfs_queue.emplace(v, v_distance);
      });
    }
  };

  perform_bfs(source_bfs_queue, *_initial_source_side_nodes);
  perform_bfs(sink_bfs_queue, *_initial_sink_side_nodes);

  return max_distance;
}

std::size_t PiercingHeuristic::compute_max_num_piercing_nodes(const NodeWeight side_weight) {
  if (!_ctx.bulk_piercing) {
    return 1;
  }

  BulkPiercingContext &bp_ctx = bulk_piercing_context();
  if (++bp_ctx.num_rounds <= _ctx.bulk_piercing_round_threshold) {
    return 1;
  }

  bp_ctx.current_weight_goal *= _ctx.bulk_piercing_shrinking_factor;
  bp_ctx.current_weight_goal_remaining += bp_ctx.current_weight_goal;

  const NodeWeight added_weight =
      side_weight - (bp_ctx.initial_side_weight + bp_ctx.weight_added_so_far);
  bp_ctx.weight_added_so_far += added_weight;
  bp_ctx.current_weight_goal_remaining -= added_weight;

  const double speed =
      bp_ctx.weight_added_so_far / static_cast<double>(bp_ctx.total_bulk_piercing_nodes);
  if (bp_ctx.current_weight_goal_remaining <= speed) {
    return 1;
  }

  const std::size_t estimated_num_piercing_nodes = bp_ctx.current_weight_goal_remaining / speed;
  return estimated_num_piercing_nodes;
}

PiercingHeuristic::BulkPiercingContext &PiercingHeuristic::bulk_piercing_context() {
  return _source_side ? _source_side_bulk_piercing_ctx : _sink_side_bulk_piercing_ctx;
}

} // namespace kaminpar::shm
