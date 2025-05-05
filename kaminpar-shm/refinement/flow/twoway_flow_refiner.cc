/*******************************************************************************
 * Two-way flow refiner.
 *
 * @file:   twoway_flow_refiner.h
 * @author: Daniel Salwasser
 * @date:   10.04.2025
 ******************************************************************************/
#include "kaminpar-shm/refinement/flow/twoway_flow_refiner.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <queue>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/balancer/sequential_greedy_balancer.h"
#include "kaminpar-shm/refinement/flow/max_flow/edmond_karp_algorithm.h"
#include "kaminpar-shm/refinement/flow/max_flow/fifo_preflow_push_algorithm.h"
#include "kaminpar-shm/refinement/flow/max_flow/highest_level_preflow_push_algorithm.h"
#include "kaminpar-shm/refinement/flow/max_flow/max_flow_algorithm.h"
#include "kaminpar-shm/refinement/flow/piercing/piercing_heuristic.h"
#include "kaminpar-shm/refinement/flow/util/border_region.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

class BipartitionFlowRefiner {
  SET_DEBUG(false);

  struct FlowNetwork {
    NodeID source;
    NodeID sink;

    CSRGraph graph;
    std::unordered_map<NodeID, NodeID> global_to_local_mapping;
  };

  struct Cut {
    NodeWeight weight;
    std::unordered_set<NodeID> nodes;
  };

public:
  struct Move {
    NodeID node;
    BlockID old_block;
    BlockID new_block;
  };

  struct Result {
    EdgeWeight gain;
    std::vector<Move> moves;

    Result() : gain(0) {};
    Result(EdgeWeight gain, std::vector<Move> moves) : gain(gain), moves(std::move(moves)) {};
  };

public:
  BipartitionFlowRefiner(
      const PartitionContext &p_ctx,
      const TwowayFlowRefinementContext &f_ctx,
      const PartitionedCSRGraph &p_graph,
      const CSRGraph &graph
  )
      : _p_ctx(p_ctx),
        _f_ctx(f_ctx),
        _p_graph(p_graph),
        _graph(graph) {
    switch (_f_ctx.flow_algorithm) {
    case FlowAlgorithm::EDMONDS_KARP:
      _max_flow_algorithm = std::make_unique<EdmondsKarpAlgorithm>();
      break;
    case FlowAlgorithm::FIFO_PREFLOW_PUSH:
      _max_flow_algorithm = std::make_unique<FIFOPreflowPushAlgorithm>(_f_ctx.fifo_preflow_push);
      break;
    case FlowAlgorithm::HIGHEST_LEVEL_PREFLOW_PUSH:
      _max_flow_algorithm =
          std::make_unique<HighestLevelPreflowPushAlgorithm>(_f_ctx.highest_level_preflow_push);
      break;
    }
  }

  [[nodiscard]] Result refine(const BlockID block1, const BlockID block2) {
    _block1 = block1;
    _block2 = block2;

    compute_border_regions();
    expand_border_region(_border_region1);
    expand_border_region(_border_region2);

    construct_flow_network();
    _border_region1.project(_flow_network.global_to_local_mapping);
    _border_region2.project(_flow_network.global_to_local_mapping);

    const NodeWeight total_weight = _p_graph.block_weight(block1) + _p_graph.block_weight(block2);
    const NodeWeight max_block1_weight = _p_ctx.max_block_weight(block1);
    const NodeWeight max_block2_weight = _p_ctx.max_block_weight(block2);

    _constrained_cut_value = _initial_cut_value;
    _unconstrained_cut_value = _initial_cut_value;

    std::unordered_set<NodeID> source_side_nodes{_flow_network.source};
    std::unordered_set<NodeID> sink_side_nodes{_flow_network.sink};
    TIMED_SCOPE("Initialize Max Flow Algorithm") {
      _max_flow_algorithm->initialize(_flow_network.graph);
    };

    PiercingHeuristic piercing_heuristic(
        _f_ctx.piercing, _flow_network.graph, _border_region1.nodes(), _border_region2.nodes()
    );

    DBG << "Starting refinement for block pair " << block1 << " and " << block2
        << " with an initial cut of " << _initial_cut_value;

    while (true) {
      const auto [cut_value, flow] = TIMED_SCOPE("Compute Max Flow") {
        return _max_flow_algorithm->compute_max_flow(source_side_nodes, sink_side_nodes);
      };
      DBG << "Found a cut for block pair " << block1 << " and " << block2 << " with value "
          << cut_value;

      if (cut_value >= _initial_cut_value) {
        DBG << "Cut is worse than the initial cut (" << _initial_cut_value << "); "
            << "aborting refinement for block pair " << block1 << " and " << block2;
        break;
      }

      Cut source_cut = compute_source_cut(source_side_nodes, flow);
      Cut sink_cut = compute_sink_cut(sink_side_nodes, flow);
      KASSERT(
          debug::are_terminals_disjoint(source_cut.nodes, sink_cut.nodes),
          "source and sink nodes are not disjoint",
          assert::heavy
      );

      const bool is_source_cut_balanced = source_cut.weight <= max_block1_weight &&
                                          (total_weight - source_cut.weight) <= max_block2_weight;
      if (is_source_cut_balanced) {
        DBG << "Found cut for block pair " << block1 << " and " << block2
            << " is a balanced source-side cut";
        _constrained_cut_value = cut_value;

        compute_moves(source_cut.nodes, true);
        break;
      }

      const bool is_sink_cut_balanced = sink_cut.weight <= max_block2_weight &&
                                        (total_weight - sink_cut.weight) <= max_block1_weight;
      if (is_sink_cut_balanced) {
        DBG << "Found cut for block pair " << block1 << " and " << block2
            << " is a balanced sink-side cut";
        _constrained_cut_value = cut_value;

        compute_moves(sink_cut.nodes, false);
        break;
      }

      if (_f_ctx.unconstrained) {
        rebalance(cut_value, source_cut.nodes, true);
        rebalance(cut_value, sink_cut.nodes, false);
      }

      SCOPED_TIMER("Compute Piercing Node");
      if (source_cut.weight <= sink_cut.weight) {
        DBG << "Piercing on source-side (" << source_cut.weight << "/" << max_block1_weight << ", "
            << (total_weight - source_cut.weight) << "/" << max_block2_weight << ")";

        const NodeWeight max_piercing_node_weight = max_block1_weight - source_cut.weight;
        const auto piercing_nodes = piercing_heuristic.pierce_on_source_side(
            source_cut.nodes, sink_cut.nodes, sink_side_nodes, max_piercing_node_weight
        );

        if (piercing_nodes.empty()) {
          LOG_WARNING << "Failed to find a suitable piercing node; "
                         "aborting refinement for block pair "
                      << block1 << " and " << block2;
          break;
        }

        source_side_nodes = std::move(source_cut.nodes);
        source_side_nodes.insert(piercing_nodes.begin(), piercing_nodes.end());
      } else {
        DBG << "Piercing on sink-side (" << sink_cut.weight << "/" << max_block2_weight << ", "
            << (total_weight - sink_cut.weight) << "/" << max_block1_weight << ")";

        const NodeWeight max_piercing_node_weight = max_block2_weight - sink_cut.weight;
        const auto piercing_nodes = piercing_heuristic.pierce_on_sink_side(
            sink_cut.nodes, source_cut.nodes, source_side_nodes, max_piercing_node_weight
        );

        if (piercing_nodes.empty()) {
          LOG_WARNING << "Failed to find a suitable piercing node; "
                         "aborting refinement for block pair "
                      << block1 << " and " << block2;
          break;
        }

        sink_side_nodes = std::move(sink_cut.nodes);
        sink_side_nodes.insert(piercing_nodes.begin(), piercing_nodes.end());
      }

      KASSERT(
          debug::are_terminals_disjoint(source_cut.nodes, sink_cut.nodes),
          "source and sink nodes are not disjoint",
          assert::heavy
      );
    }

    if (_unconstrained_cut_value < _constrained_cut_value) {
      const EdgeWeight gain = _initial_cut_value - _unconstrained_cut_value;
      return Result(gain, std::move(_unconstrained_moves));
    } else {
      const EdgeWeight gain = _initial_cut_value - _constrained_cut_value;
      return Result(gain, std::move(_constrained_moves));
    }
  }

private:
  void compute_border_regions() {
    SCOPED_TIMER("Compute Border Regions");

    const BlockID block1 = _block1;
    const BlockID block2 = _block2;

    const NodeWeight max_border_region_weight1 =
        (1 + _f_ctx.border_region_scaling_factor * _p_ctx.epsilon()) *
            _p_ctx.perfectly_balanced_block_weight(block2) -
        _p_graph.block_weight(block2);
    _border_region1.reset(block1, max_border_region_weight1);

    const NodeWeight max_border_region_weight2 =
        (1 + _f_ctx.border_region_scaling_factor * _p_ctx.epsilon()) *
            _p_ctx.perfectly_balanced_block_weight(block1) -
        _p_graph.block_weight(block1);
    _border_region2.reset(block2, max_border_region_weight2);

    EdgeWeight cut_value = 0;
    for (NodeID u : _graph.nodes()) {
      if (_p_graph.block(u) != block1) {
        continue;
      }

      const NodeWeight u_weight = _graph.node_weight(u);
      if (!_border_region1.fits(u_weight)) {
        continue;
      }

      bool is_border_region_node = false;
      _graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        if (_p_graph.block(v) != block2) {
          return;
        }

        if (_border_region2.contains(v)) {
          is_border_region_node = true;

          cut_value += w;
          return;
        }

        const NodeWeight v_weight = _graph.node_weight(v);
        if (_border_region2.fits(v_weight)) {
          is_border_region_node = true;
          _border_region2.insert(v, v_weight);

          cut_value += w;
        }
      });

      if (is_border_region_node) {
        _border_region1.insert(u, u_weight);
      }
    }

    _initial_cut_value = cut_value;
  }

  void expand_border_region(BorderRegion &border_region) const {
    SCOPED_TIMER("Expand Border Region");

    std::queue<std::pair<NodeID, NodeID>> bfs_queue;
    for (const NodeID u : border_region.nodes()) {
      bfs_queue.emplace(u, 0);
    }

    const BlockID block = border_region.block();
    while (!bfs_queue.empty()) {
      const auto [u, u_distance] = bfs_queue.front();
      bfs_queue.pop();

      _graph.adjacent_nodes(u, [&](const NodeID v) {
        if (_p_graph.block(v) != block || border_region.contains(v)) {
          return;
        }

        const NodeWeight v_weight = _graph.node_weight(v);
        if (border_region.fits(v_weight)) {
          border_region.insert(v, v_weight);

          if (u_distance < _f_ctx.max_border_distance) {
            bfs_queue.emplace(v, u_distance + 1);
          }
        }
      });
    }
  }

  void construct_flow_network() {
    SCOPED_TIMER("Construct Flow Network");

    constexpr NodeID kSource = 0;
    constexpr NodeID kSink = 1;
    constexpr NodeID kFirstNodeID = 2;

    NodeID cur_node = kFirstNodeID;

    std::unordered_map<NodeID, NodeID> global_to_local_mapping;
    for (const BorderRegion &border_region :
         {std::cref(_border_region1), std::cref(_border_region2)}) {
      for (const NodeID u : border_region.nodes()) {
        global_to_local_mapping.emplace(u, cur_node++);
      }
    }

    const NodeID num_nodes = 2 + _border_region1.num_nodes() + _border_region2.num_nodes();
    StaticArray<EdgeID> nodes(num_nodes + 1, static_array::noinit);
    StaticArray<NodeWeight> node_weights(num_nodes, static_array::noinit);

    cur_node = kFirstNodeID;
    for (BlockID terminal = 0; terminal < 2; ++terminal) {
      const BorderRegion &border_region = (terminal == 0) ? _border_region1 : _border_region2;

      NodeWeight border_region_weight = 0;
      EdgeID num_terminal_edges = 0;
      for (const NodeID u : border_region.nodes()) {
        EdgeID num_neighbors = 0;
        _graph.adjacent_nodes(u, [&](const NodeID v) {
          num_neighbors += global_to_local_mapping.contains(v) ? 1 : 0;
        });

        const bool has_non_border_region_neighbor = num_neighbors != _graph.degree(u);
        if (has_non_border_region_neighbor) { // Node has an edge to its corresponding terminal
          num_neighbors += 1;
          num_terminal_edges += 1;
        }

        const NodeWeight u_weight = _graph.node_weight(u);
        nodes[cur_node + 1] = num_neighbors;
        node_weights[cur_node] = u_weight;

        border_region_weight += u_weight;
        cur_node += 1;
      }

      nodes[terminal + 1] = num_terminal_edges;
      node_weights[terminal] = _p_graph.block_weight(border_region.block()) - border_region_weight;
    }
    KASSERT(cur_node == num_nodes);

    nodes[0] = 0;
    std::partial_sum(nodes.begin(), nodes.end(), nodes.begin());

    const EdgeID num_edges = nodes.back();
    StaticArray<NodeID> edges(num_edges, static_array::noinit);
    StaticArray<EdgeWeight> edge_weights(num_edges, static_array::noinit);

    cur_node = kFirstNodeID;
    EdgeID cur_edge = nodes[kFirstNodeID];
    EdgeID cur_source_edge = 0;
    for (BlockID terminal = 0; terminal < 2; ++terminal) {
      const BorderRegion &border_region = (terminal == 0) ? _border_region1 : _border_region2;

      for (const NodeID u : border_region.nodes()) {
        EdgeWeight terminal_edge_weight = 0;
        _graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
          if (auto it = global_to_local_mapping.find(v); it != global_to_local_mapping.end()) {
            const NodeID v_local = it->second;

            edges[cur_edge] = v_local;
            edge_weights[cur_edge] = w;
            cur_edge += 1;
          } else {
            terminal_edge_weight += w;
          }
        });

        const NodeID u_degree = cur_edge - nodes[cur_node];
        const bool has_non_border_region_neighbor = u_degree != _graph.degree(u);
        if (has_non_border_region_neighbor) { // Connect node to its corresponding terminal
          edges[cur_edge] = terminal;
          edge_weights[cur_edge] = terminal_edge_weight;
          cur_edge += 1;

          edges[cur_source_edge] = cur_node;
          edge_weights[cur_source_edge] = terminal_edge_weight;
          cur_source_edge += 1;
        }

        cur_node += 1;
      }
    }
    KASSERT(cur_node == num_nodes);
    KASSERT(cur_edge == num_edges);
    KASSERT(cur_source_edge == nodes[kFirstNodeID]);

    CSRGraph graph(
        CSRGraph::seq(),
        std::move(nodes),
        std::move(edges),
        std::move(node_weights),
        std::move(edge_weights)
    );
    KASSERT(debug::validate_graph(graph), "constructed invalid flow network", assert::heavy);

    _flow_network =
        FlowNetwork(kSource, kSink, std::move(graph), std::move(global_to_local_mapping));
  }

  Cut compute_source_cut(
      const std::unordered_set<NodeID> &sources, std::span<const EdgeWeight> flow
  ) const {
    return compute_cut(sources, flow, true);
  }

  Cut compute_sink_cut(const std::unordered_set<NodeID> &sinks, std::span<const EdgeWeight> flow)
      const {
    return compute_cut(sinks, flow, false);
  }

  Cut compute_cut(
      const std::unordered_set<NodeID> &terminals,
      std::span<const EdgeWeight> flow,
      const bool source_side
  ) const {
    SCOPED_TIMER("Compute Reachable Nodes");

    const CSRGraph &graph = _flow_network.graph;

    NodeWeight cut_weight = 0;
    std::unordered_set<NodeID> cut_nodes;

    std::queue<NodeID> bfs_queue;
    for (const NodeID terminal : terminals) {
      cut_weight += graph.node_weight(terminal);
      cut_nodes.insert(terminal);
      bfs_queue.push(terminal);
    }

    while (!bfs_queue.empty()) {
      const NodeID u = bfs_queue.front();
      bfs_queue.pop();

      graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
        if (cut_nodes.contains(v)) {
          return;
        }

        const EdgeWeight e_flow = flow[e];
        const bool has_residual_capacity = source_side ? (e_flow < c) : (-e_flow < c);
        if (has_residual_capacity) {
          cut_weight += graph.node_weight(v);
          cut_nodes.insert(v);
          bfs_queue.push(v);
        }
      });
    }

    return Cut(cut_weight, std::move(cut_nodes));
  }

  void compute_moves(const std::unordered_set<NodeID> &cut_nodes, bool source_side) {
    SCOPED_TIMER("Compute Moves");

    _constrained_moves.clear();

    const BlockID block = source_side ? _block1 : _block2;
    const BlockID other_block = source_side ? _block2 : _block1;

    const bool swap_artificial_node = source_side ? !cut_nodes.contains(_flow_network.source)
                                                  : !cut_nodes.contains(_flow_network.sink);
    const bool swap_other_artificial_node = source_side ? cut_nodes.contains(_flow_network.sink)
                                                        : cut_nodes.contains(_flow_network.source);

    const std::unordered_set<NodeID> &initial_cut_nodes =
        source_side ? _border_region1.nodes() : _border_region2.nodes();

    const std::unordered_map<NodeID, NodeID> &mapping = _flow_network.global_to_local_mapping;
    for (const NodeID u : _graph.nodes()) {
      if (auto it = mapping.find(u); it != mapping.end()) {
        const NodeID u_local = it->second;

        const BlockID old_block = initial_cut_nodes.contains(u) ? block : other_block;
        const BlockID new_block = cut_nodes.contains(u_local) ? block : other_block;

        if (old_block != new_block) {
          _constrained_moves.emplace_back(u, old_block, new_block);
        }

        continue;
      }

      const BlockID u_block = _p_graph.block(u);
      if (u_block == block) {
        if (swap_artificial_node) {
          _constrained_moves.emplace_back(u, block, other_block);
        }
      } else if (u_block == other_block) {
        if (swap_other_artificial_node) {
          _constrained_moves.emplace_back(u, other_block, block);
        }
      }
    }
  }

  void rebalance(
      const EdgeWeight cut_value,
      const std::unordered_set<NodeID> &cut_nodes,
      const bool source_side
  ) {
    SCOPED_TIMER("Rebalancing");

    PartitionedCSRGraph p_graph = copy_partitioned_graph(cut_nodes, source_side);
    const auto [rebalanced, gain] = _balancer.balance(p_graph, _graph, _p_ctx.max_block_weights());
    if (!rebalanced) {
      return;
    }

    const EdgeWeight rebalanced_cut_value = cut_value - gain;
    DBG << "Rebalanced imbalanced " << (source_side ? "source-side" : "sink-side")
        << " cut with resulting value " << rebalanced_cut_value;

    if (rebalanced_cut_value < _unconstrained_cut_value) {
      _unconstrained_cut_value = rebalanced_cut_value;
      _unconstrained_moves.clear();

      for (const NodeID u : _graph.nodes()) {
        const BlockID old_block = _p_graph.block(u);
        const BlockID new_block = p_graph.block(u);

        if (old_block != new_block) {
          _unconstrained_moves.emplace_back(u, old_block, new_block);
        }
      }
    }
  }

  PartitionedCSRGraph copy_partitioned_graph(
      const std::unordered_set<NodeID> &cut_nodes, const bool source_side
  ) const {
    PartitionedCSRGraph p_graph(PartitionedCSRGraph::seq(), _graph, _p_graph.k());

    const BlockID block = source_side ? _block1 : _block2;
    const BlockID other_block = source_side ? _block2 : _block1;

    const bool swap_artificial_node = source_side ? !cut_nodes.contains(_flow_network.source)
                                                  : !cut_nodes.contains(_flow_network.sink);
    const bool swap_other_artificial_node = source_side ? cut_nodes.contains(_flow_network.sink)
                                                        : cut_nodes.contains(_flow_network.source);

    const std::unordered_map<NodeID, NodeID> &mapping = _flow_network.global_to_local_mapping;
    for (const NodeID u : _graph.nodes()) {
      if (auto it = mapping.find(u); it != mapping.end()) {
        const NodeID u_local = it->second;
        const BlockID u_block = cut_nodes.contains(u_local) ? block : other_block;

        p_graph.set_block(u, u_block);
        continue;
      }

      const BlockID u_block = _p_graph.block(u);
      if (u_block == block) {
        if (swap_artificial_node) {
          p_graph.set_block(u, other_block);
          continue;
        }
      } else if (u_block == other_block) {
        if (swap_other_artificial_node) {
          p_graph.set_block(u, block);
          continue;
        }
      }

      p_graph.set_block(u, u_block);
    }

    return p_graph;
  }

private:
  const PartitionContext &_p_ctx;
  const TwowayFlowRefinementContext &_f_ctx;

  const PartitionedCSRGraph &_p_graph;
  const CSRGraph &_graph;

  std::unique_ptr<MaxFlowAlgorithm> _max_flow_algorithm;
  SequentialGreedyBalancerImpl<PartitionedCSRGraph, CSRGraph> _balancer;

  BlockID _block1;
  BlockID _block2;
  EdgeWeight _initial_cut_value;

  BorderRegion _border_region1;
  BorderRegion _border_region2;
  FlowNetwork _flow_network;

  EdgeWeight _constrained_cut_value;
  std::vector<Move> _constrained_moves;

  EdgeWeight _unconstrained_cut_value;
  std::vector<Move> _unconstrained_moves;
};

class SequentialBlockPairScheduler {
  SET_DEBUG(false);

  using Move = BipartitionFlowRefiner::Move;

public:
  SequentialBlockPairScheduler(const TwowayFlowRefinementContext &f_ctx) : _f_ctx(f_ctx) {}

  bool refine(PartitionedCSRGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx) {
    _p_graph = &p_graph;
    _graph = &graph;

    // Since the timers have a significant running time overhead, we disable them usually.
    IF_NOT_DBG DISABLE_TIMERS();

    activate_all_blocks();

    BipartitionFlowRefiner refiner(p_ctx, _f_ctx, p_graph, graph);

    std::size_t num_round = 0;
    bool found_improvement = false;
    EdgeWeight prev_cut_value = metrics::edge_cut_seq(p_graph);
    while (true) {
      num_round += 1;
      DBG << "Starting round " << num_round;

      EdgeWeight cut_value = prev_cut_value;
      while (!_block_pairs.empty()) {
        const auto [block1, block2] = _block_pairs.front();
        _block_pairs.pop();

        DBG << "Scheduling block pair " << block1 << " and " << block2;
        auto [gain, moves] = refiner.refine(block1, block2);

        DBG << "Found balanced cut for bock pair " << block1 << " and " << block2 << " with gain "
            << gain << " (" << cut_value << " -> " << (cut_value - gain) << ")";

        if (gain > 0) {
          cut_value -= gain;

          apply_moves(moves);
          KASSERT(
              metrics::is_balanced(p_graph, p_ctx),
              "Computed an imbalanced move sequence",
              assert::heavy
          );

          _active_blocks[block1] = true;
          _active_blocks[block2] = true;
        }
      }

      const EdgeWeight round_gain = prev_cut_value - cut_value;
      if (round_gain > 0) {
        found_improvement = true;
      }

      const double relative_improvement = round_gain / static_cast<double>(prev_cut_value);
      DBG << "Finished round with a relative improvement of " << relative_improvement;

      if (num_round == _f_ctx.max_num_rounds ||
          relative_improvement < _f_ctx.min_round_improvement_factor) {
        break;
      }

      activate_blocks();
      prev_cut_value = cut_value;
    }

    IF_NOT_DBG ENABLE_TIMERS();

    return found_improvement;
  }

private:
  void activate_all_blocks() {
    SCOPED_TIMER("Activate Blocks");

    if (_active_blocks.size() < _p_graph->k()) {
      _active_blocks.resize(_p_graph->k(), static_array::noinit);
    }

    _block_pairs = {}; // std::queue is missing a clear function
    for (BlockID block2 = 0, k = _p_graph->k(); block2 < k; ++block2) {
      for (BlockID block1 = 0; block1 < block2; ++block1) {
        _block_pairs.emplace(block1, block2);
      }
    }

    std::fill(_active_blocks.begin(), _active_blocks.end(), false);
  }

  void activate_blocks() {
    SCOPED_TIMER("Activate Blocks");

    for (BlockID block2 = 0, k = _p_graph->k(); block2 < k; ++block2) {
      if (_active_blocks[block2]) {
        for (BlockID block1 = 0; block1 < block2; ++block1) {
          if (_active_blocks[block1]) {
            _block_pairs.emplace(block1, block2);
          }
        }
      }
    }

    std::fill(_active_blocks.begin(), _active_blocks.end(), false);
  }

  void apply_moves(const std::vector<Move> &moves) {
    SCOPED_TIMER("Apply Moves");

    for (const Move &move : moves) {
      KASSERT(
          _p_graph->block(move.node) == move.old_block,
          "Move sequence contains invalid old block ids",
          assert::heavy
      );

      _p_graph->set_block(move.node, move.new_block);
    }
  }

private:
  const TwowayFlowRefinementContext &_f_ctx;

  PartitionedCSRGraph *_p_graph;
  const CSRGraph *_graph;

  StaticArray<bool> _active_blocks;
  std::queue<std::pair<BlockID, BlockID>> _block_pairs;
};

class ParallelBlockPairScheduler {
  SET_DEBUG(false);

  using Move = BipartitionFlowRefiner::Move;

public:
  ParallelBlockPairScheduler(const TwowayFlowRefinementContext &f_ctx) : _f_ctx(f_ctx) {}

  bool refine(PartitionedCSRGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx) {
    _p_graph = &p_graph;
    _graph = &graph;

    // Since timers are not multi-threaded, we disable them during parallel refinement.
    DISABLE_TIMERS();

    activate_all_blocks();

    const EdgeWeight initial_cut_value = metrics::edge_cut_seq(p_graph);
    auto refiner_ets = tbb::enumerable_thread_specific<BipartitionFlowRefiner>([&]() {
      return BipartitionFlowRefiner(p_ctx, _f_ctx, p_graph, graph);
    });

    bool found_improvement = false;
    std::size_t num_round = 0;
    EdgeWeight prev_cut_value = initial_cut_value;
    while (true) {
      num_round += 1;
      DBG << "Starting round " << num_round;

      EdgeWeight cut_value = prev_cut_value;
      tbb::parallel_for<BlockID>(0, _block_pairs.size(), [&](const BlockID i) {
        const auto [block1, block2] = _block_pairs[i];
        auto &refiner = refiner_ets.local();

        DBG << "Scheduling block pair " << block1 << " and " << block2;
        auto [expected_gain, moves] = refiner.refine(block1, block2);

        if (expected_gain <= 0) {
          return;
        }

        const std::unique_lock lock(_apply_moves_mutex);
        apply_moves(moves);

        const bool imbalance_conflict = !metrics::is_balanced(p_graph, p_ctx);
        if (imbalance_conflict) {
          DBG << "Block pair " << block1 << " and " << block2 << " has an imbalance conflict";
          revert_moves(moves);
          return;
        }

        const EdgeWeight new_cut_value = metrics::edge_cut_seq(p_graph);
        const EdgeWeight gain = cut_value - new_cut_value;
        DBG << "Found balanced cut for bock pair " << block1 << " and " << block2 << " with gain "
            << gain << " (" << cut_value << " -> " << new_cut_value << ")";

        if (gain <= 0) {
          revert_moves(moves);
          return;
        }

        cut_value = new_cut_value;
        _active_blocks[block1] = true;
        _active_blocks[block2] = true;
      });

      const EdgeWeight gain = prev_cut_value - cut_value;
      if (gain > 0) {
        found_improvement = true;
      }

      const double relative_improvement = gain / static_cast<double>(prev_cut_value);
      DBG << "Finished round with a relative improvement of " << relative_improvement;

      if (num_round == _f_ctx.max_num_rounds ||
          relative_improvement < _f_ctx.min_round_improvement_factor) {
        break;
      }

      activate_blocks();
      prev_cut_value = cut_value;
    }

    ENABLE_TIMERS();

    return found_improvement;
  }

private:
  void activate_all_blocks() {
    if (_active_blocks.size() < _p_graph->k()) {
      _active_blocks.resize(_p_graph->k(), static_array::noinit);
    }

    _block_pairs.clear();
    for (BlockID block2 = 0, k = _p_graph->k(); block2 < k; ++block2) {
      for (BlockID block1 = 0; block1 < block2; ++block1) {
        _block_pairs.emplace_back(block1, block2);
      }
    }

    std::fill(_active_blocks.begin(), _active_blocks.end(), false);
  }

  void activate_blocks() {
    _block_pairs.clear();

    for (BlockID block2 = 0, k = _p_graph->k(); block2 < k; ++block2) {
      if (_active_blocks[block2]) {
        for (BlockID block1 = 0; block1 < block2; ++block1) {
          if (_active_blocks[block1]) {
            _block_pairs.emplace_back(block1, block2);
          }
        }
      }
    }

    std::fill(_active_blocks.begin(), _active_blocks.end(), false);
  }

  void apply_moves(std::vector<Move> &moves) {
    for (Move &move : moves) {
      const NodeID u = move.node;

      // Remove all nodes from the move sequence that are not in their expected block.
      // Use the old block variable to mark the move as such, which is used during reverting.
      const BlockID invalid_block_conflict = _p_graph->block(u) != move.old_block;
      if (invalid_block_conflict) {
        move.old_block = kInvalidBlockID;
        continue;
      }

      _p_graph->set_block(u, move.new_block);
    }
  }

  void revert_moves(const std::vector<Move> &moves) {
    for (const Move &move : moves) {
      const BlockID old_block = move.old_block;

      // If the node was not in its expected block, it has not been moved.
      // Thus, the move must not be reverted.
      const BlockID invalid_block_conflict = old_block == kInvalidBlockID;
      if (invalid_block_conflict) {
        continue;
      }

      _p_graph->set_block(move.node, old_block);
    }
  }

private:
  const TwowayFlowRefinementContext &_f_ctx;

  PartitionedCSRGraph *_p_graph;
  const CSRGraph *_graph;

  StaticArray<bool> _active_blocks;
  std::vector<std::pair<BlockID, BlockID>> _block_pairs;
  std::mutex _apply_moves_mutex;
};

TwowayFlowRefiner::TwowayFlowRefiner(const TwowayFlowRefinementContext &f_ctx) : _f_ctx(f_ctx) {}

TwowayFlowRefiner::~TwowayFlowRefiner() = default;

std::string TwowayFlowRefiner::name() const {
  return "Two-Way Flow Refinement";
}

void TwowayFlowRefiner::initialize([[maybe_unused]] const PartitionedGraph &p_graph) {}

bool TwowayFlowRefiner::refine(
    PartitionedGraph &p_graph, [[maybe_unused]] const PartitionContext &p_ctx
) {
  return reified(
      p_graph,
      [&](const auto &csr_graph) {
        // The bipartition refiner works with PartitionedCSRGraph instead of PartitionedGraph.
        // Intead of copying the partition, use a span to access the partition.
        StaticArray<BlockID> &partition = p_graph.raw_partition();
        StaticArray<BlockID> partition_span(partition.size(), partition.data());

        StaticArray<BlockWeight> &block_weights = p_graph.raw_block_weights();
        StaticArray<BlockWeight> block_weights_span(block_weights.size(), block_weights.data());

        PartitionedCSRGraph p_csr_graph(
            csr_graph, p_graph.k(), std::move(partition_span), std::move(block_weights_span)
        );
        return refine(p_csr_graph, csr_graph, p_ctx);
      },
      [&]([[maybe_unused]] const auto &compressed_graph) {
        LOG_WARNING << "Cannot refine a compressed graph using the two-way flow refiner.";
        return false;
      }
  );
}

bool TwowayFlowRefiner::refine(
    PartitionedCSRGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx
) {
  SCOPED_TIMER("Two-Way Flow Refinement");
  SCOPED_HEAP_PROFILER("Two-Way Flow Refinement");

  if (_f_ctx.parallel_scheduling) {
    ParallelBlockPairScheduler scheduler(_f_ctx);
    return scheduler.refine(p_graph, graph, p_ctx);
  } else {
    SequentialBlockPairScheduler scheduler(_f_ctx);
    return scheduler.refine(p_graph, graph, p_ctx);
  }
}

} // namespace kaminpar::shm
