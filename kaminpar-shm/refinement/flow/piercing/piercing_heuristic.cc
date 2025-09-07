#include "kaminpar-shm/refinement/flow/piercing/piercing_heuristic.h"

namespace kaminpar::shm {

PiercingHeuristic::PiercingHeuristic(const PiercingHeuristicContext &ctx, const BlockID num_blocks)
    : _ph_ctx(ctx),
      _num_blocks(num_blocks),
      _random(random::thread_independent_seeding) {}

void PiercingHeuristic::initialize(
    const BorderRegion &border_region,
    const FlowNetwork &flow_network,
    const NodeWeight max_source_side_weight,
    const NodeWeight max_sink_side_weight
) {
  _border_region = &border_region;
  _flow_network = &flow_network;

  if (_ph_ctx.deterministic) {
    _random.reinit(
        Random::get_seed() + (border_region.block1() * _num_blocks + border_region.block2())
    );
  }

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
    const bool source_side, const NodeID node, const bool unreachable
) {
  const NodeWeight multiplier = source_side ? -1 : 1;
  const NodeID distance = std::max<NodeWeight>(multiplier * _distance[node], 0);

  PiercingNodeCandidatesBuckets &candidates_buckets =
      source_side ? (unreachable ? _source_unreachable_candidates_buckets
                                 : _source_reachable_candidates_buckets)
                  : (unreachable ? _sink_unreachable_candidates_buckets
                                 : _sink_reachable_candidates_buckets);
  candidates_buckets.add_candidate(node, distance);
}

std::span<const NodeID> PiercingHeuristic::find_piercing_nodes(
    const bool source_side,
    const bool has_unreachable_nodes,
    const NodeStatus &cut_status,
    const Marker<> &reachable_oracle,
    const NodeWeight side_weight,
    const NodeWeight max_weight
) {
  _piercing_nodes.clear();

  add_piercing_nodes(source_side, kUnreachableTag, cut_status, reachable_oracle, max_weight, 1);
  if (!_piercing_nodes.empty()) {
    return _piercing_nodes;
  }

  if (has_unreachable_nodes) {
    const NodeID num_reclassified_nodes =
        reclassify_reachable_candidates(source_side, cut_status, reachable_oracle, max_weight);

    if (num_reclassified_nodes > 0) {
      add_piercing_nodes(source_side, kUnreachableTag, cut_status, reachable_oracle, max_weight, 1);

      KASSERT(!_piercing_nodes.empty());
      return _piercing_nodes;
    }
  }

  const std::size_t max_piercing_nodes = compute_max_num_piercing_nodes(source_side, side_weight);
  add_piercing_nodes(
      source_side, kReachableTag, cut_status, reachable_oracle, max_weight, max_piercing_nodes
  );

  if (_ph_ctx.bulk_piercing) {
    BulkPiercingContext &bp_ctx = bulk_piercing_context(source_side);
    bp_ctx.total_bulk_piercing_nodes += _piercing_nodes.size();
  }

  if (_ph_ctx.fallback_heuristic && _piercing_nodes.empty()) {
    employ_fallback_heuristic(source_side, cut_status, max_weight);
  }

  return _piercing_nodes;
}

void PiercingHeuristic::add_piercing_nodes(
    const bool source_side,
    const bool unreachable_candidates,
    const NodeStatus &cut_status,
    const Marker<> &reachable_oracle,
    const NodeWeight max_weight,
    const NodeID max_num_piercing_nodes
) {
  PiercingNodeCandidatesBuckets &candidates_buckets =
      source_side ? (unreachable_candidates ? _source_unreachable_candidates_buckets
                                            : _source_reachable_candidates_buckets)
                  : (unreachable_candidates ? _sink_unreachable_candidates_buckets
                                            : _sink_reachable_candidates_buckets);

  const std::int64_t max_distance = candidates_buckets.max_occupied_bucket();
  const std::int64_t min_distance = candidates_buckets.min_occupied_bucket();

  const CSRGraph &graph = _flow_network->graph;
  NodeWeight cur_weight = 0;

  for (std::int64_t distance = max_distance; distance >= min_distance; --distance) {
    PiercingNodeCandidatesBucket &bucket = candidates_buckets.bucket(distance);

    if (_ph_ctx.deterministic) {
      constexpr bool kFilterOnlyNondeterministicRange = true;
      bucket.filter<kFilterOnlyNondeterministicRange>([&](const NodeID u) {
        return cut_status.is_terminal(u);
      });
      bucket.sort();
    }

    while (!bucket.empty()) {
      const std::size_t size = bucket.size();
      const std::size_t random_id = _random.random_index(0, size);

      const NodeID u = bucket.remove(random_id);
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
}

NodeID PiercingHeuristic::reclassify_reachable_candidates(
    const bool source_side,
    const NodeStatus &cut_status,
    const Marker<> &reachable_oracle,
    const NodeWeight max_node_weight
) {
  PiercingNodeCandidatesBuckets &candidates_buckets =
      source_side ? _source_reachable_candidates_buckets : _sink_reachable_candidates_buckets;

  const std::int64_t max_distance = candidates_buckets.max_occupied_bucket();
  const std::int64_t min_distance = candidates_buckets.min_occupied_bucket();

  const CSRGraph &graph = _flow_network->graph;
  NodeID num_moved = 0;

  for (std::int64_t distance = max_distance; distance >= min_distance; --distance) {
    PiercingNodeCandidatesBucket &candidates = candidates_buckets.bucket(distance);
    candidates.filter([&](const NodeID u) {
      if (cut_status.is_terminal(u) || graph.node_weight(u) > max_node_weight) {
        return true;
      }

      if (!reachable_oracle.get(u)) {
        add_piercing_node_candidate(source_side, u, kUnreachableTag);

        num_moved += 1;
        return true;
      }

      return false;
    });
  }

  return num_moved;
}

void PiercingHeuristic::employ_fallback_heuristic(
    const bool source_side, const NodeStatus &cut_status, const NodeWeight max_node_weight
) {
  KASSERT(_piercing_nodes.empty());

  const CSRGraph &graph = _flow_network->graph;
  const NodeWeight multiplier = source_side ? -1 : 1;

  NodeWeight cur_distance = -1;
  for (const NodeID u : graph.nodes()) {
    if (cut_status.is_terminal(u)) {
      continue;
    }

    const NodeWeight u_weight = graph.node_weight(u);
    if (u_weight > max_node_weight) {
      continue;
    }

    const NodeWeight distance = std::max<NodeWeight>(multiplier * _distance[u], 0);
    if (distance > cur_distance) {
      cur_distance = distance;

      _piercing_nodes.clear();
      _piercing_nodes.push_back(u);
    } else if (distance == cur_distance) {
      _piercing_nodes.push_back(u);
    }
  }

  if (!_piercing_nodes.empty()) {
    const std::size_t idx = _random.random_index(0, _piercing_nodes.size());
    std::swap(_piercing_nodes[0], _piercing_nodes[idx]);
    _piercing_nodes.resize(1);
  }
}

std::size_t PiercingHeuristic::compute_max_num_piercing_nodes(
    const bool source_side, const NodeWeight side_weight
) {
  if (!_ph_ctx.bulk_piercing) {
    return 1;
  }

  BulkPiercingContext &bp_ctx = bulk_piercing_context(source_side);
  if (++bp_ctx.num_rounds < _ph_ctx.bulk_piercing_round_threshold) {
    return 1;
  }

  bp_ctx.current_weight_goal *= _ph_ctx.bulk_piercing_shrinking_factor;
  bp_ctx.current_weight_goal_remaining += bp_ctx.current_weight_goal;

  NodeWeight added_weight = side_weight - (bp_ctx.initial_side_weight + bp_ctx.weight_added_so_far);
  bp_ctx.weight_added_so_far += added_weight;
  bp_ctx.current_weight_goal_remaining -= added_weight;

  double speed = bp_ctx.weight_added_so_far / static_cast<double>(bp_ctx.total_bulk_piercing_nodes);
  if (bp_ctx.current_weight_goal_remaining <= speed) {
    return 1;
  }

  std::size_t estimated_num_piercing_nodes = bp_ctx.current_weight_goal_remaining / speed;
  return estimated_num_piercing_nodes;
}

PiercingHeuristic::BulkPiercingContext &PiercingHeuristic::bulk_piercing_context(bool source_side) {
  return source_side ? _source_side_bulk_piercing_ctx : _sink_side_bulk_piercing_ctx;
}

void PiercingHeuristic::free() {
  _piercing_nodes.clear();
  _piercing_nodes.shrink_to_fit();

  _bfs_runner.free();
  _distance.free();

  _source_reachable_candidates_buckets.free();
  _source_unreachable_candidates_buckets.free();

  _sink_reachable_candidates_buckets.free();
  _sink_unreachable_candidates_buckets.free();
}

} // namespace kaminpar::shm
