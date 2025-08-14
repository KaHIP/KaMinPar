#include "kaminpar-shm/refinement/flow/piercing/piercing_heuristic.h"

#include <limits>

namespace kaminpar::shm {

PiercingHeuristic::PiercingHeuristic(const PiercingHeuristicContext &ctx) : _ph_ctx(ctx) {}

void PiercingHeuristic::initialize(
    const CSRGraph &graph,
    const NodeWeight source_side_weight,
    const NodeWeight sink_side_weight,
    const NodeWeight total_weight,
    const NodeWeight max_source_side_weight,
    const NodeWeight max_sink_side_weight
) {
  _graph = &graph;

  if (_initial_side_status.size() < graph.n()) {
    _initial_side_status.resize(graph.n(), static_array::noinit);
  }
  std::fill_n(_initial_side_status.begin(), graph.n(), kUnknown);

  if (_distance.size() < _graph->n()) {
    _distance.resize(_graph->n(), static_array::noinit);
  }
  std::fill_n(_distance.begin(), _graph->n(), kInvalidNodeID);

  const NodeWeight max_total_weight = max_source_side_weight + max_sink_side_weight;
  _source_side_bulk_piercing_ctx.initialize(
      source_side_weight, total_weight, max_source_side_weight, max_total_weight
  );
  _sink_side_bulk_piercing_ctx.initialize(
      sink_side_weight, total_weight, max_sink_side_weight, max_total_weight
  );
}

void PiercingHeuristic::set_initial_node_side(const NodeID node, const bool source_side) {
  _initial_side_status[node] = source_side ? kInitialSourceSide : kInitialSinkSide;
}

void PiercingHeuristic::reset(const bool source_side) {
  _source_side = source_side;

  _reachable_candidates_buckets.reset();
  _unreachable_candidates_buckets.reset();
}

void PiercingHeuristic::compute_distances() {
  _source_side_bfs_runner.reset();
  _sink_side_bfs_runner.reset();

  for (const NodeID u : _graph->nodes()) {
    if (_initial_side_status[u] != kInitialSourceSide) {
      continue;
    }

    bool has_cut_edge = false;
    _graph->adjacent_nodes(u, [&](const NodeID v) {
      if (_initial_side_status[v] != kInitialSinkSide) {
        return;
      }

      has_cut_edge = true;
      if (_distance[v] == kInvalidNodeID) {
        _distance[v] = 1;
        _sink_side_bfs_runner.add_seed(v);
      }
    });

    if (has_cut_edge) {
      _distance[u] = 1;
      _source_side_bfs_runner.add_seed(u);
    }
  }

  NodeID max_distance = 1;
  const auto perform_bfs = [&](auto &bfs_runner, const auto side_status) {
    bfs_runner.perform(1, [&](const NodeID u, const NodeID u_distance, auto &queue) {
      const NodeID v_distance = u_distance + 1;

      _graph->adjacent_nodes(u, [&](const NodeID v) {
        if (_distance[v] != kInvalidNodeID || _initial_side_status[v] != side_status) {
          return;
        }

        max_distance = std::max(max_distance, v_distance);
        _distance[v] = v_distance;

        queue.push_back(v);
      });
    });
  };

  perform_bfs(_source_side_bfs_runner, kInitialSourceSide);
  perform_bfs(_sink_side_bfs_runner, kInitialSinkSide);

  _reachable_candidates_buckets.initialize(max_distance);
  _unreachable_candidates_buckets.initialize(max_distance);
}

void PiercingHeuristic::add_piercing_node_candidate(const NodeID node, const bool reachable) {
  const std::uint8_t current_side = _source_side ? kInitialSourceSide : kInitialSinkSide;
  const NodeID distance = (_initial_side_status[node] == current_side) ? _distance[node] : 0;

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
  const auto add_piercing_nodes = [&](auto &candidates_buckets, const auto max_num_piercing_nodes) {
    const std::int64_t max_bucket = _unreachable_candidates_buckets.max_occupied_bucket();
    const std::int64_t min_bucket = _unreachable_candidates_buckets.min_occupied_bucket();

    for (std::int64_t bucket = max_bucket; bucket >= min_bucket; --bucket) {
      if (_ph_ctx.deterministic) {
        candidates_buckets.sort(bucket);
      }

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
      _ph_ctx.pierce_all_viable ? std::numeric_limits<std::size_t>::max() : max_num_piercing_nodes
  );
  add_piercing_nodes(_reachable_candidates_buckets, max_num_piercing_nodes);

  if (_ph_ctx.bulk_piercing) {
    BulkPiercingContext &bp_ctx = bulk_piercing_context();
    bp_ctx.total_bulk_piercing_nodes += _piercing_nodes.size();
  }

  if (_ph_ctx.fallback_heuristic && _piercing_nodes.empty()) {
    const std::uint8_t current_initial_side = _source_side ? kInitialSourceSide : kInitialSinkSide;

    const std::uint8_t cut_side = _source_side ? NodeStatus::kSource : NodeStatus::kSink;
    const std::uint8_t other_cut_side = _source_side ? NodeStatus::kSink : NodeStatus::kSource;

    NodeID cur_piercing_node = kInvalidNodeID;
    NodeID cur_distance = kInvalidNodeID;
    for (const NodeID u : _graph->nodes()) {
      if (cut_status.has_status(u, cut_side) || terminal_status.has_status(u, other_cut_side)) {
        continue;
      }

      const NodeWeight u_weight = _graph->node_weight(u);
      if (u_weight > max_weight) {
        continue;
      }

      const NodeID distance = (_initial_side_status[u] == current_initial_side) ? _distance[u] : 0;
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

std::size_t PiercingHeuristic::compute_max_num_piercing_nodes(const NodeWeight side_weight) {
  if (!_ph_ctx.bulk_piercing) {
    return 1;
  }

  BulkPiercingContext &bp_ctx = bulk_piercing_context();
  if (++bp_ctx.num_rounds <= _ph_ctx.bulk_piercing_round_threshold) {
    return 1;
  }

  bp_ctx.current_weight_goal *= _ph_ctx.bulk_piercing_shrinking_factor;
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
