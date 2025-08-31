#include "kaminpar-shm/refinement/flow/piercing/piercing_heuristic.h"

#include <utility>

#include "kaminpar-common/random.h"

namespace kaminpar::shm {

PiercingHeuristic::PiercingHeuristic(const PiercingHeuristicContext &ctx) : _ph_ctx(ctx) {}

void PiercingHeuristic::initialize(
    const BorderRegion &border_region,
    const FlowNetwork &flow_network,
    const NodeWeight max_source_side_weight,
    const NodeWeight max_sink_side_weight
) {
  _border_region = &border_region;
  _flow_network = &flow_network;

  const NodeWeight initial_source_side_weight = flow_network.graph.node_weight(flow_network.source);
  const NodeWeight initial_sink_side_weight = flow_network.graph.node_weight(flow_network.sink);
  const NodeWeight total_weight = flow_network.graph.total_node_weight();
  const NodeWeight max_total_weight = max_source_side_weight + max_sink_side_weight;
  _source_side_bulk_piercing_ctx.initialize(
      initial_source_side_weight, total_weight, max_source_side_weight, max_total_weight
  );
  _sink_side_bulk_piercing_ctx.initialize(
      initial_sink_side_weight, total_weight, max_sink_side_weight, max_total_weight
  );

  compute_distances();
}

void PiercingHeuristic::compute_distances() {
  const NodeID source = _flow_network->source;
  const NodeID sink = _flow_network->sink;
  const CSRGraph &graph = _flow_network->graph;

  if (_distance.size() < graph.n()) {
    _distance.resize(graph.n(), static_array::noinit);
  }

  NodeWeight max_dist_source = 1;
  NodeWeight max_dist_sink = 1;
  if (_ph_ctx.determine_distance_from_cut) {
    std::fill_n(_distance.data(), graph.n(), kInvalidNodeWeight);

    _bfs_runner.reset();
    for (const NodeID u : _border_region->initial_nodes_region1()) {
      const NodeID u_local = _flow_network->global_to_local_mapping.get(u);
      _distance[u_local] = -1;
      _bfs_runner.add_seed(u_local);
    }
    for (const NodeID u : _border_region->initial_nodes_region2()) {
      const NodeID u_local = _flow_network->global_to_local_mapping.get(u);
      _distance[u_local] = 1;
      _bfs_runner.add_seed(u_local);
    }

    _bfs_runner.perform(1, [&](const NodeID u, const NodeID u_distance, auto &queue) {
      const NodeWeight v_distance = u_distance + 1;

      graph.adjacent_nodes(u, [&](const NodeID v) {
        if (v == source || v == sink || _distance[v] != kInvalidNodeWeight) {
          return;
        }

        const NodeID v_global = _flow_network->local_to_global_mapping.get(v);
        const bool source_side = _border_region->region1_contains(v_global);

        max_dist_source = source_side ? std::max(max_dist_source, v_distance) : max_dist_source;
        max_dist_sink = source_side ? max_dist_sink : std::max(max_dist_sink, v_distance);

        _distance[v] = source_side ? -v_distance : v_distance;
        queue.push_back(v);
      });
    });

    _distance[source] = -(max_dist_source + 1);
    _distance[sink] = max_dist_sink + 1;
  } else {
    std::fill_n(_distance.data(), graph.n(), 0);
  }

  _source_reachable_candidates_buckets.initialize(max_dist_source);
  _source_unreachable_candidates_buckets.initialize(max_dist_source);

  _sink_reachable_candidates_buckets.initialize(max_dist_sink);
  _sink_unreachable_candidates_buckets.initialize(max_dist_sink);
}

void PiercingHeuristic::add_piercing_node_candidate(
    const bool source_side, const NodeID node, const bool reachable
) {
  const NodeID distance = std::max<NodeWeight>((source_side ? -1 : 1) * _distance[node], 0);

  if (source_side) {
    if (reachable) {
      _source_reachable_candidates_buckets.add_candidate(node, distance);
    } else {
      _source_unreachable_candidates_buckets.add_candidate(node, distance);
    }
  } else {
    if (reachable) {
      _sink_reachable_candidates_buckets.add_candidate(node, distance);
    } else {
      _sink_unreachable_candidates_buckets.add_candidate(node, distance);
    }
  }
}

std::span<const NodeID> PiercingHeuristic::find_piercing_nodes(
    const bool source_side,
    const NodeStatus &cut_status,
    const Marker<> &reachable_oracle,
    const NodeWeight side_weight,
    const NodeWeight max_weight
) {
  _piercing_nodes.clear();

  const CSRGraph &graph = _flow_network->graph;
  Random &random = Random::instance();

  NodeWeight cur_weight = 0;
  const auto add_piercing_nodes = [&](auto &candidates_buckets,
                                      const auto max_num_piercing_nodes,
                                      const bool unreachable_candidates) {
    const std::int64_t max_bucket = candidates_buckets.max_occupied_bucket();
    const std::int64_t min_bucket = candidates_buckets.min_occupied_bucket();

    for (std::int64_t bucket = max_bucket; bucket >= min_bucket; --bucket) {
      ScalableVector<NodeID> &candidates = candidates_buckets.candidates(bucket);

      if (_ph_ctx.deterministic) {
        std::sort(candidates.begin(), candidates.end());
      }

      while (!candidates.empty()) {
        const std::size_t size = candidates.size();
        const std::size_t idx = random.random_index(0, size);
        std::swap(candidates[idx], candidates[size - 1]);

        const NodeID u = candidates.back();
        candidates.pop_back();

        if (cut_status.is_terminal(u)) {
          continue;
        }

        const NodeWeight u_weight = graph.node_weight(u);
        if (cur_weight + u_weight > max_weight) {
          continue;
        }

        if (unreachable_candidates && reachable_oracle.get(u)) {
          add_piercing_node_candidate(source_side, u, kReachableTag);
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

  add_piercing_nodes(
      source_side ? _source_unreachable_candidates_buckets : _sink_unreachable_candidates_buckets,
      1ul,
      true
  );

  if (_piercing_nodes.empty()) {
    const std::size_t max_num_piercing_nodes =
        compute_max_num_piercing_nodes(source_side, side_weight);

    add_piercing_nodes(
        source_side ? _source_reachable_candidates_buckets : _sink_reachable_candidates_buckets,
        max_num_piercing_nodes,
        false
    );

    if (_ph_ctx.bulk_piercing) {
      BulkPiercingContext &bp_ctx =
          source_side ? _source_side_bulk_piercing_ctx : _sink_side_bulk_piercing_ctx;
      bp_ctx.total_bulk_piercing_nodes += _piercing_nodes.size();
    }

    if (_ph_ctx.fallback_heuristic && _piercing_nodes.empty()) {
      NodeWeight cur_distance = -1;

      for (const NodeID u : graph.nodes()) {
        if (cut_status.is_terminal(u)) {
          continue;
        }

        const NodeWeight u_weight = graph.node_weight(u);
        if (u_weight > max_weight) {
          continue;
        }

        const NodeWeight distance = std::max<NodeWeight>((source_side ? -1 : 1) * _distance[u], 0);
        if (distance > cur_distance) {
          cur_distance = distance;

          _piercing_nodes.clear();
          _piercing_nodes.push_back(u);
        } else if (distance == cur_distance) {
          _piercing_nodes.push_back(u);
        }
      }

      if (!_piercing_nodes.empty()) {
        const std::size_t idx = random.random_index(0, _piercing_nodes.size());
        std::swap(_piercing_nodes[0], _piercing_nodes[idx]);
        _piercing_nodes.resize(1);
      }
    }
  }

  return _piercing_nodes;
}

std::size_t PiercingHeuristic::compute_max_num_piercing_nodes(
    const bool source_side, const NodeWeight side_weight
) {
  if (!_ph_ctx.bulk_piercing) {
    return 1;
  }

  BulkPiercingContext &bp_ctx =
      source_side ? _source_side_bulk_piercing_ctx : _sink_side_bulk_piercing_ctx;
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

} // namespace kaminpar::shm
