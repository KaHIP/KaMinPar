#include "kaminpar-shm/refinement/flow/flow_cutter/flow_cutter.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

FlowCutter::FlowCutter(
    const PartitionContext &p_ctx, const FlowCutterContext &fc_ctx, const bool run_sequentially
)
    : _p_ctx(p_ctx),
      _fc_ctx(fc_ctx),
      _piercing_heuristic(fc_ctx.piercing, p_ctx.k) {
  if (run_sequentially) {
    _max_flow_algorithm = std::make_unique<PreflowPushAlgorithm>(fc_ctx.flow);
  } else {
    _max_flow_algorithm = std::make_unique<ParallelPreflowPushAlgorithm>(fc_ctx.flow);
  }
};

FlowCutter::Result
FlowCutter::compute_cut(const BorderRegion &border_region, const FlowNetwork &flow_network) {
  SCOPED_TIMER("Run FlowCutter");

  DBG << "Starting refinement for block pair " << border_region.block1() << " and "
      << border_region.block2() << " with an initial cut of " << flow_network.cut_value;

  _source_side_border_nodes.clear();
  _source_side_border_nodes.push_back(flow_network.source);

  _sink_side_border_nodes.clear();
  _sink_side_border_nodes.push_back(flow_network.sink);

  _source_reachable_nodes_marker.resize(flow_network.graph.n());
  _sink_reachable_nodes_marker.resize(flow_network.graph.n());

  _source_side_piercing_node_candidates_marker.reset();
  _sink_side_piercing_node_candidates_marker.reset();

  _source_side_piercing_node_candidates_marker.resize(flow_network.graph.n());
  _sink_side_piercing_node_candidates_marker.resize(flow_network.graph.n());

  const NodeWeight max_source_side_weight = _p_ctx.max_block_weight(border_region.block1());
  const NodeWeight max_sink_side_weight = _p_ctx.max_block_weight(border_region.block2());

  TIMED_SCOPE("Initialize Piercing Heuristic") {
    _piercing_heuristic.initialize(
        border_region, flow_network, max_source_side_weight, max_sink_side_weight
    );
  };

  TIMED_SCOPE("Initialize Max-Flow Algorithm") {
    _max_flow_algorithm->initialize(
        flow_network.graph, flow_network.reverse_edges, flow_network.source, flow_network.sink
    );
  };

  const NodeWeight total_weight = flow_network.graph.total_node_weight();
  NodeWeight prev_source_side_weight = flow_network.graph.node_weight(flow_network.source);
  NodeWeight prev_sink_side_weight = flow_network.graph.node_weight(flow_network.sink);

  bool augmenting_path_available_from_piercing = true;
  bool pierced_on_source_side;

  EdgeWeight cut_value;
  std::span<const EdgeWeight> flow;
  while (true) {
    if (augmenting_path_available_from_piercing) {
      TIMED_SCOPE("Compute Max Flow") {
        const auto result = _max_flow_algorithm->compute_max_preflow();
        cut_value = result.flow_value;
        flow = result.flow;
      };

      DBG << "Found a cut for block pair " << border_region.block1() << " and "
          << border_region.block2() << " with value " << cut_value;

      if (cut_value > flow_network.cut_value) {
        DBG << "Cut is worse than the initial cut (" << flow_network.cut_value << "); "
            << "aborting refinement for block pair " << border_region.block1() << " and "
            << border_region.block2();
        return Result::Empty();
      }

      constexpr bool kCollectExcessNodes = true;
      derive_source_side_cut<kCollectExcessNodes>(flow_network, flow);
      derive_sink_side_cut(flow_network, flow);
    } else {
      if (pierced_on_source_side) {
        constexpr bool kCollectExcessNodes = false;
        derive_source_side_cut<kCollectExcessNodes>(flow_network, flow);
      } else {
        derive_sink_side_cut(flow_network, flow);
      }
    }

    KASSERT(
        std::none_of(
            _source_reachable_nodes.begin(),
            _source_reachable_nodes.end(),
            [&](const NodeID node) { return _sink_reachable_nodes_marker.get(node); }
        ),
        "source and sink reachable nodes are not disjoint",
        assert::heavy
    );
    KASSERT(
        std::none_of(
            _sink_reachable_nodes.begin(),
            _sink_reachable_nodes.end(),
            [&](const NodeID node) { return _source_reachable_nodes_marker.get(node); }
        ),
        "source and sink reachable nodes are not disjoint",
        assert::heavy
    );

    EdgeWeight source_side_weight = prev_source_side_weight + _source_reachable_weight;
    EdgeWeight sink_side_weight = prev_sink_side_weight + _sink_reachable_weight;

    const bool is_source_cut_balanced = source_side_weight <= max_source_side_weight &&
                                        (total_weight - source_side_weight) <= max_sink_side_weight;
    if (is_source_cut_balanced) {
      DBG << "Found cut for block pair " << border_region.block1() << " and "
          << border_region.block2() << " is a balanced source-side cut";

      const EdgeWeight gain = flow_network.cut_value - cut_value;
      const bool improve_balance = std::max(source_side_weight, total_weight - source_side_weight) <
                                   std::max(flow_network.block1_weight, flow_network.block2_weight);

      _max_flow_algorithm->add_sources(_source_reachable_nodes);
      compute_moves(kSourceTag, border_region, flow_network);

      return Result(gain, improve_balance, _moves);
    }

    const bool is_sink_cut_balanced = sink_side_weight <= max_sink_side_weight &&
                                      (total_weight - sink_side_weight) <= max_source_side_weight;
    if (is_sink_cut_balanced) {
      DBG << "Found cut for block pair " << border_region.block1() << " and "
          << border_region.block2() << " is a balanced sink-side cut";

      const EdgeWeight gain = flow_network.cut_value - cut_value;
      const bool improve_balance = std::max(total_weight - sink_side_weight, sink_side_weight) <
                                   std::max(flow_network.block1_weight, flow_network.block2_weight);

      _max_flow_algorithm->add_sinks(_sink_reachable_nodes);
      compute_moves(kSinkTag, border_region, flow_network);

      return Result(gain, improve_balance, _moves);
    }

    if (source_side_weight <= sink_side_weight) {
      DBG << "Piercing on source-side (" << source_side_weight << "/" << max_source_side_weight
          << ", " << (total_weight - source_side_weight) << "/" << max_sink_side_weight << ")";

      TIMED_SCOPE("Update Max-Flow Algorithm State") {
        _max_flow_algorithm->add_sources(_source_reachable_nodes);
      };

      update_border_nodes(
          kSourceTag, flow_network, _source_reachable_nodes, _source_side_border_nodes
      );

      const NodeWeight max_piercing_node_weight = max_source_side_weight - source_side_weight;
      const auto piercing_nodes = TIMED_SCOPE("Compute Piercing Nodes") {
        const EdgeWeight reachable_weight = source_side_weight + sink_side_weight;
        const bool has_unreachable_nodes = (total_weight - reachable_weight) > 0;
        return _piercing_heuristic.find_piercing_nodes(
            kSourceTag,
            has_unreachable_nodes,
            _max_flow_algorithm->node_status(),
            _sink_reachable_nodes_marker,
            source_side_weight,
            max_piercing_node_weight
        );
      };

      if (piercing_nodes.empty()) {
        DBG << "Failed to find a suitable piercing node; "
               "aborting refinement for block pair "
            << border_region.block1() << " and " << border_region.block2();
        return Result::Empty();
      }

      TIMED_SCOPE("Update Max-Flow Algorithm State") {
        _max_flow_algorithm->pierce_nodes(kSourceTag, piercing_nodes);
      };

      TIMED_SCOPE("Update Border Nodes") {
        augmenting_path_available_from_piercing = false;
        _source_side_border_nodes.clear();

        for (const NodeID piercing_node : piercing_nodes) {
          augmenting_path_available_from_piercing |=
              _sink_reachable_nodes_marker.get(piercing_node);

          _source_side_border_nodes.push_back(piercing_node);
          source_side_weight += flow_network.graph.node_weight(piercing_node);
        }
      };

      prev_source_side_weight = source_side_weight;
      pierced_on_source_side = true;
    } else {
      DBG << "Piercing on sink-side (" << sink_side_weight << "/" << max_sink_side_weight << ", "
          << (total_weight - sink_side_weight) << "/" << max_source_side_weight << ")";

      TIMED_SCOPE("Update Max-Flow Algorithm State") {
        _max_flow_algorithm->add_sinks(_sink_reachable_nodes);
      };

      update_border_nodes(kSinkTag, flow_network, _sink_reachable_nodes, _sink_side_border_nodes);

      const NodeWeight max_piercing_node_weight = max_sink_side_weight - sink_side_weight;
      const auto piercing_nodes = TIMED_SCOPE("Compute Piercing Nodes") {
        const EdgeWeight reachable_weight = source_side_weight + sink_side_weight;
        const bool has_unreachable_nodes = (total_weight - reachable_weight) > 0;
        return _piercing_heuristic.find_piercing_nodes(
            kSinkTag,
            has_unreachable_nodes,
            _max_flow_algorithm->node_status(),
            _source_reachable_nodes_marker,
            sink_side_weight,
            max_piercing_node_weight
        );
      };

      if (piercing_nodes.empty()) {
        DBG << "Failed to find a suitable piercing node; "
               "aborting refinement for block pair "
            << border_region.block1() << " and " << border_region.block2();
        return Result::Empty();
      }

      TIMED_SCOPE("Update Max-Flow Algorithm State") {
        _max_flow_algorithm->pierce_nodes(kSinkTag, piercing_nodes);
      };

      TIMED_SCOPE("Update Border Nodes") {
        augmenting_path_available_from_piercing = false;
        _sink_side_border_nodes.clear();

        for (const NodeID piercing_node : piercing_nodes) {
          augmenting_path_available_from_piercing |=
              _source_reachable_nodes_marker.get(piercing_node);

          _sink_side_border_nodes.push_back(piercing_node);
          sink_side_weight += flow_network.graph.node_weight(piercing_node);
        }
      };

      prev_sink_side_weight = sink_side_weight;
      pierced_on_source_side = false;
    }

    if (time_limit_exceeded()) {
      return Result::TimeLimitExceeded();
    }
  }
}

void FlowCutter::free() {
  _max_flow_algorithm->free();

  _source_side_border_nodes.clear();
  _source_side_border_nodes.shrink_to_fit();

  _sink_side_border_nodes.clear();
  _sink_side_border_nodes.shrink_to_fit();

  _source_reachable_nodes.clear();
  _source_reachable_nodes.shrink_to_fit();

  _sink_reachable_nodes.clear();
  _sink_reachable_nodes.shrink_to_fit();

  _bfs_runner.free();
  _source_reachable_nodes_marker.free();
  _sink_reachable_nodes_marker.free();

  _piercing_heuristic.free();
  _source_side_piercing_node_candidates_marker.free();
  _sink_side_piercing_node_candidates_marker.free();

  _moves.clear();
  _moves.shrink_to_fit();
}

template <bool kCollectExcessNodes>
void FlowCutter::derive_source_side_cut(
    const FlowNetwork &flow_network, std::span<const EdgeWeight> flow
) {
  SCOPED_TIMER("Derive Cut");
  const CSRGraph &graph = flow_network.graph;

  _bfs_runner.reset();
  _source_reachable_nodes.clear();
  _source_reachable_nodes_marker.reset();

  const NodeStatus &node_status = _max_flow_algorithm->node_status();
  for (const NodeID border_node : _source_side_border_nodes) {
    KASSERT(node_status.is_source(border_node));

    _bfs_runner.add_seed(border_node);
    _source_reachable_nodes.push_back(border_node);
  }

  NodeWeight source_reachable_weight = 0;
  if constexpr (kCollectExcessNodes) {
    for (const NodeID excess_node : _max_flow_algorithm->excess_nodes()) {
      KASSERT(!node_status.is_source(excess_node));
      KASSERT(!_source_reachable_nodes_marker.get(excess_node));

      _bfs_runner.add_seed(excess_node);

      _source_reachable_nodes.push_back(excess_node);
      _source_reachable_nodes_marker.set(excess_node);

      source_reachable_weight += graph.node_weight(excess_node);
    }
  }

  _bfs_runner.perform([&](const NodeID u, auto &queue) {
    graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
      if (node_status.is_terminal(v) || _source_reachable_nodes_marker.get(v)) {
        return;
      }

      const EdgeWeight e_flow = flow[e];
      const bool has_residual_capacity = e_flow < c;
      if (has_residual_capacity) {
        queue.push_back(v);

        _source_reachable_nodes.push_back(v);
        _source_reachable_nodes_marker.set(v);

        source_reachable_weight += graph.node_weight(v);
      }
    });
  });

  _source_reachable_weight = source_reachable_weight;
}

void FlowCutter::derive_sink_side_cut(
    const FlowNetwork &flow_network, std::span<const EdgeWeight> flow
) {
  SCOPED_TIMER("Derive Cut");
  const CSRGraph &graph = flow_network.graph;

  _bfs_runner.reset();
  _sink_reachable_nodes.clear();
  _sink_reachable_nodes_marker.reset();

  const NodeStatus &node_status = _max_flow_algorithm->node_status();
  for (const NodeID border_node : _sink_side_border_nodes) {
    KASSERT(node_status.is_sink(border_node));

    _bfs_runner.add_seed(border_node);
    _sink_reachable_nodes.push_back(border_node);
  }

  NodeWeight sink_reachable_weight = 0;
  _bfs_runner.perform([&](const NodeID u, auto &queue) {
    graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
      if (node_status.is_terminal(v) || _sink_reachable_nodes_marker.get(v)) {
        return;
      }

      const EdgeWeight e_flow = flow[e];
      const bool has_residual_capacity = -e_flow < c;
      if (has_residual_capacity) {
        queue.push_back(v);

        _sink_reachable_nodes.push_back(v);
        _sink_reachable_nodes_marker.set(v);

        sink_reachable_weight += graph.node_weight(v);
      }
    });
  });

  _sink_reachable_weight = sink_reachable_weight;
}

void FlowCutter::update_border_nodes(
    const bool source_side,
    const FlowNetwork &flow_network,
    const std::span<const NodeID> reachable_nodes,
    ScalableVector<NodeID> &border_nodes
) {
  SCOPED_TIMER("Update Border Nodes");

  border_nodes.clear();

  Marker<> &piercing_marker = source_side ? _source_side_piercing_node_candidates_marker
                                          : _sink_side_piercing_node_candidates_marker;
  const CSRGraph &graph = flow_network.graph;

  const NodeStatus &node_status = _max_flow_algorithm->node_status();
  const std::uint8_t other_side_status = source_side ? NodeStatus::kSink : NodeStatus::kSource;
  for (const NodeID u : reachable_nodes) {
    bool is_border_node = false;

    graph.adjacent_nodes(u, [&](const NodeID v) {
      if (node_status.is_terminal(v)) {
        return;
      }

      is_border_node = true;
      if (!piercing_marker.get(v)) {
        piercing_marker.set(v);

        const bool unreachable = !node_status.has_status(v, other_side_status);
        _piercing_heuristic.add_piercing_node_candidate(source_side, v, unreachable);
      }
    });

    if (is_border_node) {
      border_nodes.push_back(u);
    }
  }
}

void FlowCutter::compute_moves(
    const bool source_side, const BorderRegion &border_region, const FlowNetwork &flow_network
) {
  SCOPED_TIMER("Compute Moves");

  const NodeStatus &node_status = _max_flow_algorithm->node_status();
  const BlockID block1 = border_region.block1();
  const BlockID block2 = border_region.block2();

  const std::uint8_t side_status = source_side ? NodeStatus::kSource : NodeStatus::kSink;
  const BlockID side = source_side ? block1 : block2;
  const BlockID other_side = source_side ? block2 : block1;

  _moves.clear();
  for (const auto &[u, u_local] : flow_network.global_to_local_mapping.entries()) {
    const BlockID old_block = border_region.region1_contains(u) ? block1 : block2;
    const BlockID new_block = node_status.has_status(u_local, side_status) ? side : other_side;

    if (old_block != new_block) {
      _moves.emplace_back(u, old_block, new_block);
    }
  }
}

} // namespace kaminpar::shm
