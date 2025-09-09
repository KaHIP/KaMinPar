#include "kaminpar-shm/refinement/flow/flow_cutter/hyper_flow_cutter.h"

#ifdef KAMINPAR_WHFC_FOUND
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

HyperFlowCutter::HyperFlowCutter(
    const PartitionContext &p_ctx, const FlowCutterContext &fc_ctx, const bool run_sequentially
)
    : _p_ctx(p_ctx),
      _fc_ctx(fc_ctx),
      _run_sequentially(run_sequentially),
      _hypergraph(),
      _sequential_flow_cutter(_hypergraph, Random::get_seed(), fc_ctx.piercing.deterministic),
      _parallel_flow_cutter(_hypergraph, Random::get_seed(), fc_ctx.piercing.deterministic) {
  _sequential_flow_cutter.timer.active = false;
  _sequential_flow_cutter.find_most_balanced = false;
  _sequential_flow_cutter.forceSequential(true);
  _sequential_flow_cutter.setBulkPiercing(fc_ctx.piercing.bulk_piercing);

  _parallel_flow_cutter.timer.active = false;
  _parallel_flow_cutter.find_most_balanced = false;
  _parallel_flow_cutter.forceSequential(false);
  _parallel_flow_cutter.setBulkPiercing(fc_ctx.piercing.bulk_piercing);
}

HyperFlowCutter::Result
HyperFlowCutter::compute_cut(const BorderRegion &border_region, const FlowNetwork &flow_network) {
  SCOPED_TIMER("Run WHFC");

  initialize(flow_network);
  if (_run_sequentially) {
    run_flow_cutter(_sequential_flow_cutter, border_region, flow_network);
  } else {
    run_flow_cutter(_parallel_flow_cutter, border_region, flow_network);
  }

  if (time_limit_exceeded()) {
    return Result::time_limit();
  }

  return Result(_gain, _improve_balance, _moves);
}

void HyperFlowCutter::initialize(const FlowNetwork &flow_network) {
  TIMED_SCOPE("Construct Hypergraph") {
    _hypergraph.reinitialize(flow_network.graph.n());

    const CSRGraph &graph = flow_network.graph;
    for (const NodeID u : graph.nodes()) {
      _hypergraph.nodeWeight(whfc::Node(u)) = whfc::NodeWeight(graph.node_weight(u));

      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight c) {
        if (u < v) {
          _hypergraph.startHyperedge(whfc::Flow(c));
          _hypergraph.addPin(whfc::Node(u));
          _hypergraph.addPin(whfc::Node(v));
        }
      });
    }

    _hypergraph.finalize();
  };

  _gain = 0;
  _improve_balance = false;
  _moves.clear();
}

template <typename FlowCutter>
void HyperFlowCutter::run_flow_cutter(
    FlowCutter &flow_cutter, const BorderRegion &border_region, const FlowNetwork &flow_network
) {
  SCOPED_TIMER("Run HyperFlowCutter");

  const NodeWeight max_block1_weight = _p_ctx.max_block_weight(border_region.block1());
  const NodeWeight max_block2_weight = _p_ctx.max_block_weight(border_region.block2());
  const NodeWeight total_weight = flow_network.block1_weight + flow_network.block2_weight;

  DBG << "Starting refinement for block pair " << border_region.block1() << " and "
      << border_region.block2() << " with an initial cut of " << flow_network.cut_value;

  auto &cutter_state = flow_cutter.cs;
  const auto on_cut = [&] {
    const EdgeWeight cut_value = cutter_state.flow_algo.flow_value;
    DBG << "Found a cut for block pair " << border_region.block1() << " and "
        << border_region.block2() << " with value " << cut_value;

    if (cutter_state.isBalanced()) {
      DBG << "Found cut for block pair " << border_region.block1() << " and "
          << border_region.block2() << " is a balanced cut";
      return true;
    }

    if (cutter_state.side_to_pierce == 0) {
      const EdgeWeight source_side_weight = cutter_state.source_reachable_weight;
      DBG << "Piercing on source-side (" << source_side_weight << "/" << max_block1_weight << ", "
          << (total_weight - source_side_weight) << "/" << max_block2_weight << ")";
    } else {
      const EdgeWeight sink_side_weight = cutter_state.target_reachable_weight;
      DBG << "Piercing on sink-side (" << sink_side_weight << "/" << max_block2_weight << ", "
          << (total_weight - sink_side_weight) << "/" << max_block1_weight << ")";
    }

    if (time_limit_exceeded()) {
      return false;
    }

    return true;
  };

  flow_cutter.cs.setMaxBlockWeight(0, std::max(flow_network.block1_weight, max_block1_weight));
  flow_cutter.cs.setMaxBlockWeight(1, std::max(flow_network.block2_weight, max_block2_weight));

  flow_cutter.reset();
  flow_cutter.setFlowBound(flow_network.cut_value);

  if (_fc_ctx.piercing.determine_distance_from_cut) {
    compute_distances(border_region, flow_network, flow_cutter.cs.border_nodes.distance);
    flow_cutter.cs.border_nodes.updateMaxDistance();
  }

  const bool success = flow_cutter.enumerateCutsUntilBalancedOrFlowBoundExceeded(
      whfc::Node(flow_network.source), whfc::Node(flow_network.sink), on_cut
  );

  const EdgeWeight cut_value = cutter_state.flow_algo.flow_value;
  DBG << "Found a cut for block pair " << border_region.block1() << " and "
      << border_region.block2() << " with value " << cut_value;

  if (success) {
    _gain = flow_network.cut_value - cut_value;
    _improve_balance =
        std::max<NodeWeight>(cutter_state.source_weight, cutter_state.target_weight) <
        std::max(flow_network.block1_weight, flow_network.block2_weight);

    compute_moves(border_region, flow_network, cutter_state);
  } else if (cut_value > flow_network.cut_value) {
    DBG << "Cut is worse than the initial cut (" << flow_network.cut_value << "); "
        << "aborting refinement for block pair " << border_region.block1() << " and "
        << border_region.block2();
  }
}

void HyperFlowCutter::compute_distances(
    const BorderRegion &border_region,
    const FlowNetwork &flow_network,
    std::vector<whfc::HopDistance> &distances
) {
  whfc::HopDistance max_dist_source(0);
  whfc::HopDistance max_dist_sink(0);

  const NodeID source = flow_network.source;
  const NodeID sink = flow_network.sink;
  const CSRGraph &graph = flow_network.graph;

  distances.assign(graph.n(), whfc::HopDistance(0));

  _bfs_runner.reset();
  _bfs_marker.reset();
  _bfs_marker.resize(graph.n());

  for (const NodeID u : border_region.initial_nodes_region1()) {
    const NodeID u_local = flow_network.global_to_local_mapping.get(u);
    _bfs_marker.set(u_local);
    _bfs_runner.add_seed(u_local);
  }
  for (const NodeID u : border_region.initial_nodes_region2()) {
    const NodeID u_local = flow_network.global_to_local_mapping.get(u);
    _bfs_marker.set(u_local);
    _bfs_runner.add_seed(u_local);
  }

  _bfs_runner.perform(1, [&](const NodeID u, const NodeID u_distance, auto &queue) {
    const NodeID u_global = flow_network.local_to_global_mapping.get(u);
    const bool source_side = border_region.region1_contains(u_global);

    const whfc::HopDistance dist(u_distance);
    if (source_side) {
      distances[u] = -dist;
      max_dist_source = std::max(max_dist_source, dist);
    } else {
      distances[u] = dist;
      max_dist_sink = std::max(max_dist_sink, dist);
    }

    graph.adjacent_nodes(u, [&](const NodeID v) {
      if (v == source || v == sink || _bfs_marker.get(v)) {
        return;
      }

      _bfs_marker.set(v);
      queue.push_back(v);
    });
  });

  distances[source] = -(max_dist_source + 1);
  distances[sink] = max_dist_sink + 1;
}

template <typename CutterState>
void HyperFlowCutter::compute_moves(
    const BorderRegion &border_region,
    const FlowNetwork &flow_network,
    const CutterState &cutter_state
) {
  const BlockID block1 = border_region.block1();
  const BlockID block2 = border_region.block2();
  const auto &flow_algorithm = cutter_state.flow_algo;

  _moves.clear();
  for (const auto &[u, u_local] : flow_network.global_to_local_mapping.entries()) {
    const BlockID old_block = border_region.region1_contains(u) ? block1 : block2;
    const BlockID new_block = flow_algorithm.isSource(whfc::Node(u_local)) ? block1 : block2;

    if (old_block != new_block) {
      _moves.emplace_back(u, old_block, new_block);
    }
  }
}

void HyperFlowCutter::free() {
  _bfs_marker.free();
  _bfs_runner.free();

  _moves.clear();
  _moves.shrink_to_fit();
}

} // namespace kaminpar::shm

#endif
