/*******************************************************************************
 * Two-way flow refiner.
 *
 * @file:   twoway_flow_refiner.h
 * @author: Daniel Salwasser
 * @date:   10.04.2025
 ******************************************************************************/
#include "kaminpar-shm/refinement/flow/twoway_flow_refiner.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <numeric>
#include <queue>
#include <span>
#include <utility>
#include <vector>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/task.h>

#ifdef KAMINPAR_WHFC_FOUND
#include "algorithm/hyperflowcutter.h"
#include "algorithm/sequential_push_relabel.h"
#include "assert.h"
#include "datastructure/flow_hypergraph_builder.h"
#include "definitions.h"
#endif

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/flow/max_flow/edmond_karp_algorithm.h"
#include "kaminpar-shm/refinement/flow/max_flow/fifo_preflow_push_algorithm.h"
#include "kaminpar-shm/refinement/flow/max_flow/max_flow_algorithm.h"
#include "kaminpar-shm/refinement/flow/piercing/piercing_heuristic.h"
#include "kaminpar-shm/refinement/flow/rebalancer/dynamic_greedy_balancer.h"
#include "kaminpar-shm/refinement/flow/rebalancer/static_greedy_balancer.h"
#include "kaminpar-shm/refinement/flow/util/border_region.h"
#include "kaminpar-shm/refinement/flow/util/node_status.h"
#include "kaminpar-shm/refinement/flow/util/reverse_edge_index.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/marker.h"
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
    StaticArray<NodeID> reverse_edges;
    std::unordered_map<NodeID, NodeID> global_to_local_mapping;
  };

  static constexpr bool kSourceTag = true;
  static constexpr bool kSinkTag = false;

public:
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = std::chrono::time_point<Clock>;

  struct Move {
    NodeID node;
    BlockID old_block;
    BlockID new_block;
  };

  struct Result {
    bool time_limit_exceeded;

    EdgeWeight gain;
    ScalableVector<Move> moves;

    Result(bool time_limit_exceeded = false) : time_limit_exceeded(time_limit_exceeded), gain(0) {};
    Result(EdgeWeight gain, ScalableVector<Move> moves)
        : time_limit_exceeded(false),
          gain(gain),
          moves(std::move(moves)) {};

    [[nodiscard]] static Result TimeLimitExceeded() {
      return Result(true);
    }
  };

public:
  BipartitionFlowRefiner(
      const PartitionContext &p_ctx,
      const TwowayFlowRefinementContext &f_ctx,
      const PartitionedCSRGraph &p_graph,
      const CSRGraph &graph,
      const TimePoint &start_time
  )
      : _p_ctx(p_ctx),
        _f_ctx(f_ctx),
        _p_graph(p_graph),
        _graph(graph),
        _start_time(start_time),
        _partition(graph.n(), static_array::noinit),
        _block_weights(p_graph.k(), static_array::noinit),
        _piercing_nodes_candidates_marker(graph.n()),
        _piercing_heuristic(f_ctx.piercing),
        _dynamic_balancer(p_ctx.max_block_weights()),
        _source_side_balancer(p_ctx.max_block_weights()),
        _sink_side_balancer(p_ctx.max_block_weights()) {
    switch (_f_ctx.flow_algorithm) {
    case FlowAlgorithm::EDMONDS_KARP:
      _max_flow_algorithm = std::make_unique<EdmondsKarpAlgorithm>();
      break;
    case FlowAlgorithm::FIFO_PREFLOW_PUSH:
      _max_flow_algorithm = std::make_unique<FIFOPreflowPushAlgorithm>(_f_ctx.fifo_preflow_push);
      break;
    }

    if (_f_ctx.unconstrained) {
      _p_graph_rebalancing_copy =
          PartitionedCSRGraph(PartitionedCSRGraph::seq(), graph, p_graph.k());
    }
  }

  Result refine(const EdgeWeight cut_value, const BlockID block1, const BlockID block2) {
    KASSERT(block1 != block2, "Only different block pairs can be refined");
    SCOPED_TIMER("Refine Block Pair");
    _time_limit_exceeded = false;

    _block1 = block1;
    _block2 = block2;
    _global_cut_value = cut_value;
    initialize_block_data();

    compute_border_regions();
    expand_border_region(_border_region1);
    expand_border_region(_border_region2);

    construct_flow_network();
    _border_region1.project(_flow_network.global_to_local_mapping);
    _border_region2.project(_flow_network.global_to_local_mapping);

    _constrained_cut_value = _global_cut_value;
    _unconstrained_cut_value = _global_cut_value;

    DBG << "Starting refinement for block pair " << _block1 << " and " << _block2
        << " with an initial cut of " << _initial_cut_value << " (global: " << cut_value << ")";

    if (_f_ctx.unconstrained) {
      initialize_rebalancer();
    }

    if (_f_ctx.use_whfc) {
      run_hyper_flow_cutter();
    } else {
      run_flow_cutter();
    }

    if (_time_limit_exceeded) {
      return Result::TimeLimitExceeded();
    }

    if (_unconstrained_cut_value < _constrained_cut_value) {
      const EdgeWeight gain = _global_cut_value - _unconstrained_cut_value;
      return Result(gain, std::move(_unconstrained_moves));
    } else {
      const EdgeWeight gain = _global_cut_value - _constrained_cut_value;
      return Result(gain, std::move(_constrained_moves));
    }
  }

private:
  void initialize_block_data() {
    SCOPED_TIMER("Initialize Block Data");

    std::fill(_block_weights.begin(), _block_weights.end(), 0);

    for (const NodeID u : _graph.nodes()) {
      const BlockID u_block = _p_graph.block(u);
      const NodeWeight u_weight = _graph.node_weight(u);

      _partition[u] = u_block;
      _block_weights[u_block] += u_weight;
    }
  }

  void compute_border_regions() {
    SCOPED_TIMER("Compute Border Regions");

    const BlockID block1 = _block1;
    const BlockID block2 = _block2;

    const NodeWeight max_border_region_weight1 =
        (1 + _f_ctx.border_region_scaling_factor * _p_ctx.epsilon()) *
            _p_ctx.perfectly_balanced_block_weight(block2) -
        _block_weights[_block2];
    _border_region1.reset(block1, max_border_region_weight1);

    const NodeWeight max_border_region_weight2 =
        (1 + _f_ctx.border_region_scaling_factor * _p_ctx.epsilon()) *
            _p_ctx.perfectly_balanced_block_weight(block1) -
        _block_weights[_block1];
    _border_region2.reset(block2, max_border_region_weight2);

    for (const NodeID u : _graph.nodes()) {
      if (_partition[u] != block1) {
        continue;
      }

      const NodeWeight u_weight = _graph.node_weight(u);
      if (!_border_region1.fits(u_weight)) {
        continue;
      }

      bool is_border_region_node = false;
      _graph.adjacent_nodes(u, [&](const NodeID v) {
        if (_partition[v] != block2) {
          return;
        }

        if (_border_region2.contains(v)) {
          is_border_region_node = true;
          return;
        }

        const NodeWeight v_weight = _graph.node_weight(v);
        if (_border_region2.fits(v_weight)) {
          is_border_region_node = true;
          _border_region2.insert(v, v_weight);
        }
      });

      if (is_border_region_node) {
        _border_region1.insert(u, u_weight);
      }
    }
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

      const NodeID v_distance = u_distance + 1;
      _graph.adjacent_nodes(u, [&](const NodeID v) {
        if (_partition[v] != block || border_region.contains(v)) {
          return;
        }

        const NodeWeight v_weight = _graph.node_weight(v);
        if (border_region.fits(v_weight)) {
          border_region.insert(v, v_weight);

          if (v_distance <= _f_ctx.max_border_distance) {
            bfs_queue.emplace(v, v_distance);
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
    EdgeID num_source_edges = 0;
    EdgeID num_sink_edges = 0;
    for (BlockID terminal = 0; terminal < 2; ++terminal) {
      const BorderRegion &border_region = (terminal == 0) ? _border_region1 : _border_region2;

      for (const NodeID u : border_region.nodes()) {
        EdgeID num_neighbors = 0;
        bool has_source_edge = false;
        bool has_sink_edge = false;

        _graph.adjacent_nodes(u, [&](const NodeID v) {
          if (global_to_local_mapping.contains(v)) {
            num_neighbors += 1;
            return;
          }

          const BlockID v_block = _partition[v];
          if (v_block == _block1) {
            has_source_edge = true;
          } else if (v_block == _block2) {
            has_sink_edge = true;
          }
        });

        if (has_source_edge) {
          num_neighbors += 1;
          num_source_edges += 1;
        }

        if (has_sink_edge) {
          num_neighbors += 1;
          num_sink_edges += 1;
        }

        const NodeWeight u_weight = _graph.node_weight(u);
        nodes[cur_node] = num_neighbors;
        node_weights[cur_node] = u_weight;

        cur_node += 1;
      }
    }

    nodes[kSource] = num_source_edges;
    node_weights[kSource] = _block_weights[_block1] - _border_region1.weight();

    nodes[kSink] = num_sink_edges;
    node_weights[kSink] = _block_weights[_block2] - _border_region2.weight();

    nodes.back() = 0;
    std::partial_sum(nodes.begin(), nodes.end(), nodes.begin());

    const EdgeID num_edges = nodes.back();
    StaticArray<NodeID> edges(num_edges, static_array::noinit);
    StaticArray<EdgeWeight> edge_weights(num_edges, static_array::noinit);
    StaticArray<NodeID> reverse_edges(num_edges, static_array::noinit);

    EdgeWeight cut_value = 0;

    cur_node = kFirstNodeID;
    EdgeID cur_source_edge = nodes[kSource];
    EdgeID cur_sink_edge = nodes[kSink];
    for (BlockID terminal = 0; terminal < 2; ++terminal) {
      const BorderRegion &border_region = (terminal == 0) ? _border_region1 : _border_region2;

      const bool is_source_terminal = terminal == 0;
      const bool is_sink_terminal = terminal == 1;

      for (const NodeID u : border_region.nodes()) {
        const NodeID u_local = cur_node;

        bool has_source_edge = false;
        EdgeWeight source_edge_weight = 0;

        bool has_sink_edge = false;
        EdgeWeight sink_edge_weight = 0;

        EdgeID u_edge = nodes[u_local];
        _graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
          if (auto it = global_to_local_mapping.find(v); it != global_to_local_mapping.end()) {
            const NodeID v_local = it->second;
            if (u_local >= v_local) {
              return;
            }

            u_edge -= 1;
            edges[u_edge] = v_local;
            edge_weights[u_edge] = w;

            const EdgeID v_edge = --nodes[v_local];
            edges[v_edge] = u_local;
            edge_weights[v_edge] = w;

            reverse_edges[u_edge] = v_edge;
            reverse_edges[v_edge] = u_edge;

            cut_value += (is_source_terminal && _border_region2.contains(v)) ? w : 0;
            cut_value += (is_sink_terminal && _border_region1.contains(v)) ? w : 0;

            return;
          }

          const BlockID v_block = _partition[v];
          if (v_block == _block1) {
            has_source_edge = true;
            source_edge_weight += w;
            cut_value += is_sink_terminal ? w : 0;
          } else if (v_block == _block2) {
            has_sink_edge = true;
            sink_edge_weight += w;
            cut_value += is_source_terminal ? w : 0;
          }
        });

        if (has_sink_edge) {
          u_edge -= 1;
          edges[u_edge] = kSink;
          edge_weights[u_edge] = sink_edge_weight;

          cur_sink_edge -= 1;
          edges[cur_sink_edge] = cur_node;
          edge_weights[cur_sink_edge] = sink_edge_weight;

          reverse_edges[u_edge] = cur_sink_edge;
          reverse_edges[cur_sink_edge] = u_edge;
        }

        if (has_source_edge) {
          u_edge -= 1;
          edges[u_edge] = kSource;
          edge_weights[u_edge] = source_edge_weight;

          cur_source_edge -= 1;
          edges[cur_source_edge] = cur_node;
          edge_weights[cur_source_edge] = source_edge_weight;

          reverse_edges[u_edge] = cur_source_edge;
          reverse_edges[cur_source_edge] = u_edge;
        }

        nodes[cur_node] = u_edge;
        cur_node += 1;
      }
    }

    nodes[kSource] = cur_source_edge;
    nodes[kSink] = cur_sink_edge;
    CSRGraph graph(
        CSRGraph::seq(),
        std::move(nodes),
        std::move(edges),
        std::move(node_weights),
        std::move(edge_weights)
    );

    KASSERT(debug::validate_graph(graph), "constructed an invalid flow network", assert::heavy);
    KASSERT(
        debug::is_valid_reverse_edge_index(graph, reverse_edges),
        "constructed an invalid reverse edge index",
        assert::heavy
    );

    _initial_cut_value = cut_value;
    _flow_network = FlowNetwork(
        kSource,
        kSink,
        std::move(graph),
        std::move(reverse_edges),
        std::move(global_to_local_mapping)
    );
  }

  void run_hyper_flow_cutter() {
#ifdef KAMINPAR_WHFC_FOUND
    SCOPED_TIMER("Run WHFC");

    START_TIMER("Build Hypergraph");
    whfc::FlowHypergraphBuilder hypergraph;
    hypergraph.reinitialize(_flow_network.graph.n());

    for (const NodeID u : _flow_network.graph.nodes()) {
      hypergraph.nodeWeight(whfc::Node(u)) = whfc::NodeWeight(_flow_network.graph.node_weight(u));

      _flow_network.graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight c) {
        if (u >= v) {
          return;
        }

        hypergraph.startHyperedge(whfc::Flow(c));
        hypergraph.addPin(whfc::Node(u));
        hypergraph.addPin(whfc::Node(v));
      });
    }

    hypergraph.finalize();
    STOP_TIMER();

    const NodeWeight total_weight = _block_weights[_block1] + _block_weights[_block2];
    const NodeWeight max_block1_weight = _p_ctx.max_block_weight(_block1);
    const NodeWeight max_block2_weight = _p_ctx.max_block_weight(_block2);

    START_TIMER("Run HyperFlowCutter");
    whfc::HyperFlowCutter<whfc::SequentialPushRelabel> flow_cutter(hypergraph, 1);
    flow_cutter.forceSequential(true);
    flow_cutter.setBulkPiercing(_f_ctx.piercing.bulk_piercing);

    flow_cutter.setFlowBound(_initial_cut_value);
    flow_cutter.cs.setMaxBlockWeight(0, max_block1_weight);
    flow_cutter.cs.setMaxBlockWeight(1, max_block2_weight);

    const auto on_cut = [&]() {
      const EdgeWeight cut_value = flow_cutter.cs.flow_algo.flow_value;
      DBG << "Found a cut for block pair " << _block1 << " and " << _block2 << " with value "
          << cut_value;

      if (flow_cutter.cs.isBalanced()) {
        DBG << "Found cut for block pair " << _block1 << " and " << _block2 << " is a balanced cut";
        return true;
      }

      if (_f_ctx.unconstrained) {
        const EdgeWeight gain = _initial_cut_value - flow_cutter.cs.flow_algo.flow_value;
        const EdgeWeight cut_value = _global_cut_value - gain;

        if (flow_cutter.cs.source_reachable_weight > _p_ctx.max_block_weight(_block1)) {
          SCOPED_TIMER("Rebalance source-side cut");

          const auto fetch_block = [&](const NodeID u) {
            const bool is_on_source_side =
                flow_cutter.cs.flow_algo.isSourceReachable(whfc::Node(u));
            return is_on_source_side ? _block1 : _block2;
          };

          rebalance(true, cut_value, fetch_block);
        }

        if (flow_cutter.cs.target_reachable_weight > _p_ctx.max_block_weight(_block2)) {
          SCOPED_TIMER("Rebalance sink-side cut");

          const auto fetch_block = [&](const NodeID u) {
            const bool is_on_sink_side = flow_cutter.cs.flow_algo.isTargetReachable(whfc::Node(u));
            return is_on_sink_side ? _block2 : _block1;
          };

          rebalance(false, cut_value, fetch_block);
        }

        if (_f_ctx.abort_on_candidate_cut && _unconstrained_cut_value < _global_cut_value) {
          const EdgeWeight current_gain = _initial_cut_value - cut_value;
          const EdgeWeight current_cut_value = _global_cut_value - current_gain;

          if (_unconstrained_cut_value < current_cut_value) {
            return false;
          }
        }
      }

      if (flow_cutter.cs.side_to_pierce == 0) {
        const EdgeWeight source_side_weight = flow_cutter.cs.source_reachable_weight;
        DBG << "Piercing on source-side (" << source_side_weight << "/" << max_block1_weight << ", "
            << (total_weight - source_side_weight) << "/" << max_block2_weight << ")";
      } else {
        const EdgeWeight sink_side_weight = flow_cutter.cs.target_reachable_weight;
        DBG << "Piercing on sink-side (" << sink_side_weight << "/" << max_block2_weight << ", "
            << (total_weight - sink_side_weight) << "/" << max_block1_weight << ")";
      }

      if (time_limit_exceeded()) {
        _time_limit_exceeded = true;
        return false;
      }

      return true;
    };

    const bool success = flow_cutter.enumerateCutsUntilBalancedOrFlowBoundExceeded(
        whfc::Node(_flow_network.source), whfc::Node(_flow_network.sink), on_cut
    );
    STOP_TIMER();

    if (!success) {
      if (flow_cutter.cs.flow_algo.flow_value >= _initial_cut_value) {
        DBG << "Cut is worse than the initial cut (" << _initial_cut_value << "); "
            << "aborting refinement for block pair " << _block1 << " and " << _block2;
      }

      return;
    }

    const EdgeWeight gain = _initial_cut_value - flow_cutter.cs.flow_algo.flow_value;
    _constrained_cut_value = _global_cut_value - gain;

    DBG << "Found a balanced cut for block pair " << _block1 << " and " << _block2 << " with value "
        << flow_cutter.cs.flow_algo.flow_value;

    const auto fetch_block = [&](const NodeID u) {
      const bool is_on_source_side = flow_cutter.cs.flow_algo.isSource(whfc::Node(u));
      return is_on_source_side ? _block1 : _block2;
    };

    compute_moves(fetch_block);
#else
    LOG_WARNING << "WHFC is not available; skipping refinement";
#endif
  }

  void run_flow_cutter() {
    SCOPED_TIMER("Run FlowCutter");

    _node_status.initialize(_flow_network.graph.n());

    _source_side_border_nodes.clear();
    _source_side_border_nodes.push_back(_flow_network.source);

    _sink_side_border_nodes.clear();
    _sink_side_border_nodes.push_back(_flow_network.sink);

    const NodeWeight total_weight = _block_weights[_block1] + _block_weights[_block2];
    const NodeWeight max_source_side_weight = _p_ctx.max_block_weight(_block1);
    const NodeWeight max_sink_side_weight = _p_ctx.max_block_weight(_block2);

    TIMED_SCOPE("Initialize Piercing Heuristic") {
      _piercing_heuristic.initialize(
          _flow_network.graph,
          _border_region1.nodes(),
          _border_region2.nodes(),
          _flow_network.graph.node_weight(_flow_network.source),
          _flow_network.graph.node_weight(_flow_network.sink),
          total_weight,
          max_source_side_weight,
          max_sink_side_weight
      );
    };

    TIMED_SCOPE("Initialize Max-Flow Algorithm") {
      _max_flow_algorithm->initialize(
          _flow_network.graph, _flow_network.reverse_edges, _flow_network.source, _flow_network.sink
      );
    };

    NodeWeight prev_source_side_weight = _flow_network.graph.node_weight(_flow_network.source);
    NodeWeight prev_sink_side_weight = _flow_network.graph.node_weight(_flow_network.sink);
    while (true) {
      const auto [cut_value, flow] = TIMED_SCOPE("Compute Max Flow") {
        return _max_flow_algorithm->compute_max_flow();
      };
      DBG << "Found a cut for block pair " << _block1 << " and " << _block2 << " with value "
          << cut_value;

      if (cut_value >= _initial_cut_value) {
        DBG << "Cut is worse than the initial cut (" << _initial_cut_value << "); "
            << "aborting refinement for block pair " << _block1 << " and " << _block2;
        break;
      }

      const auto [source_side_weight_increase, sink_side_weight_increase] = expand_cuts(flow);
      NodeWeight source_side_weight = prev_source_side_weight + source_side_weight_increase;
      NodeWeight sink_side_weight = prev_sink_side_weight + sink_side_weight_increase;

      const bool is_source_cut_balanced =
          source_side_weight <= max_source_side_weight &&
          (total_weight - source_side_weight) <= max_sink_side_weight;
      if (is_source_cut_balanced) {
        DBG << "Found cut for block pair " << _block1 << " and " << _block2
            << " is a balanced source-side cut";

        const EdgeWeight gain = _initial_cut_value - cut_value;
        _constrained_cut_value = _global_cut_value - gain;

        const auto fetch_block = [&](const NodeID u) {
          return _node_status.is_source(u) ? _block1 : _block2;
        };

        compute_moves(fetch_block);
        break;
      }

      const bool is_sink_cut_balanced = sink_side_weight <= max_sink_side_weight &&
                                        (total_weight - sink_side_weight) <= max_source_side_weight;
      if (is_sink_cut_balanced) {
        DBG << "Found cut for block pair " << _block1 << " and " << _block2
            << " is a balanced sink-side cut";

        const EdgeWeight gain = _initial_cut_value - cut_value;
        _constrained_cut_value = _global_cut_value - gain;

        const auto fetch_block = [&](const NodeID u) {
          return _node_status.is_sink(u) ? _block2 : _block1;
        };

        compute_moves(fetch_block);
        break;
      }

      if (_f_ctx.unconstrained) {
        const EdgeWeight gain = _initial_cut_value - cut_value;
        const EdgeWeight cut_value = _global_cut_value - gain;

        if (source_side_weight > max_source_side_weight) {
          SCOPED_TIMER("Rebalance source-side cut");

          const auto fetch_block = [&](const NodeID u) {
            return _node_status.is_source(u) ? _block1 : _block2;
          };

          rebalance(kSourceTag, cut_value, fetch_block);
        }

        if (sink_side_weight > max_sink_side_weight) {
          SCOPED_TIMER("Rebalance sink-side cut");

          const auto fetch_block = [&](const NodeID u) {
            return _node_status.is_sink(u) ? _block2 : _block1;
          };

          rebalance(kSinkTag, cut_value, fetch_block);
        }

        if (_f_ctx.abort_on_candidate_cut && _unconstrained_cut_value < _global_cut_value) {
          const EdgeWeight current_gain = _initial_cut_value - cut_value;
          const EdgeWeight current_cut_value = _global_cut_value - current_gain;

          if (_unconstrained_cut_value < current_cut_value) {
            break;
          }
        }
      }

      if (source_side_weight <= sink_side_weight) {
        DBG << "Piercing on source-side (" << source_side_weight << "/" << max_source_side_weight
            << ", " << (total_weight - source_side_weight) << "/" << max_sink_side_weight << ")";

        update_border_nodes(
            kSourceTag, _source_side_border_nodes_candidates, _source_side_border_nodes
        );

        const NodeWeight max_piercing_node_weight = max_source_side_weight - source_side_weight;
        const auto piercing_nodes = TIMED_SCOPE("Compute Piercing Nodes") {
          return _piercing_heuristic.find_piercing_nodes(
              _node_status,
              _max_flow_algorithm->node_status(),
              source_side_weight,
              max_piercing_node_weight
          );
        };

        if (piercing_nodes.empty()) {
          LOG_WARNING << "Failed to find a suitable piercing node; "
                         "aborting refinement for block pair "
                      << _block1 << " and " << _block2;
          break;
        }

        TIMED_SCOPE("Update Max-Flow Algorithm State") {
          _max_flow_algorithm->add_sources(_node_status.source_nodes());
          _max_flow_algorithm->pierce_nodes(piercing_nodes, kSourceTag);
        };

        TIMED_SCOPE("Update Border Nodes") {
          for (const NodeID piercing_node : piercing_nodes) {
            source_side_weight += _flow_network.graph.node_weight(piercing_node);
            _source_side_border_nodes.push_back(piercing_node);
          }
        };

        prev_source_side_weight = source_side_weight;
      } else {
        DBG << "Piercing on sink-side (" << sink_side_weight << "/" << max_sink_side_weight << ", "
            << (total_weight - sink_side_weight) << "/" << max_source_side_weight << ")";

        update_border_nodes(kSinkTag, _sink_side_border_nodes_candidates, _sink_side_border_nodes);

        const NodeWeight max_piercing_node_weight = max_sink_side_weight - sink_side_weight;
        const auto piercing_nodes = TIMED_SCOPE("Compute Piercing Nodes") {
          return _piercing_heuristic.find_piercing_nodes(
              _node_status,
              _max_flow_algorithm->node_status(),
              sink_side_weight,
              max_piercing_node_weight
          );
        };

        if (piercing_nodes.empty()) {
          LOG_WARNING << "Failed to find a suitable piercing node; "
                         "aborting refinement for block pair "
                      << _block1 << " and " << _block2;
          break;
        }

        TIMED_SCOPE("Update Max-Flow Algorithm State") {
          _max_flow_algorithm->add_sinks(_node_status.sink_nodes());
          _max_flow_algorithm->pierce_nodes(piercing_nodes, kSinkTag);
        };

        TIMED_SCOPE("Update Border Nodes") {
          for (const NodeID piercing_node : piercing_nodes) {
            sink_side_weight += _flow_network.graph.node_weight(piercing_node);
            _sink_side_border_nodes.push_back(piercing_node);
          }
        };

        prev_sink_side_weight = sink_side_weight;
      }

      if (time_limit_exceeded()) {
        _time_limit_exceeded = true;
        break;
      }
    }
  }

  std::pair<NodeWeight, NodeWeight> expand_cuts(std::span<const EdgeWeight> flow) {
    const NodeStatus &cut_status = _max_flow_algorithm->node_status();

    _node_status.reset();
    for (const NodeID source : cut_status.source_nodes()) {
      _node_status.add_source(source);
    }
    for (const NodeID sink : cut_status.sink_nodes()) {
      _node_status.add_sink(sink);
    }

    _source_side_border_nodes_candidates.clear();
    const NodeWeight source_side_weight_increase = expand_cut(
        kSourceTag, flow, _source_side_border_nodes, _source_side_border_nodes_candidates
    );

    _sink_side_border_nodes_candidates.clear();
    const NodeWeight sink_side_weight_increase =
        expand_cut(kSinkTag, flow, _sink_side_border_nodes, _sink_side_border_nodes_candidates);

    return {source_side_weight_increase, sink_side_weight_increase};
  }

  NodeWeight expand_cut(
      const bool source_side,
      std::span<const EdgeWeight> flow,
      std::span<const NodeID> border_nodes,
      ScalableVector<NodeID> &border_nodes_candidates
  ) {
    SCOPED_TIMER("Expand Cut");
    const CSRGraph &graph = _flow_network.graph;

    std::queue<NodeID> bfs_queue;
    for (const NodeID border_node : border_nodes) {
      bfs_queue.push(border_node);
      border_nodes_candidates.push_back(border_node);
    }

    NodeWeight cut_weight_increase = 0;
    while (!bfs_queue.empty()) {
      const NodeID u = bfs_queue.front();
      bfs_queue.pop();

      graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
        if (!_node_status.is_unknown(v)) {
          return;
        }

        const EdgeWeight e_flow = flow[e];
        const bool has_residual_capacity = source_side ? (e_flow < c) : (-e_flow < c);
        if (has_residual_capacity) {
          bfs_queue.push(v);

          cut_weight_increase += graph.node_weight(v);
          border_nodes_candidates.push_back(v);

          if (source_side) {
            _node_status.add_source(v);
          } else {
            _node_status.add_sink(v);
          }
        }
      });
    }

    return cut_weight_increase;
  }

  void update_border_nodes(
      const bool source_side,
      const ScalableVector<NodeID> &potential_border_nodes,
      ScalableVector<NodeID> &border_nodes
  ) {
    SCOPED_TIMER("Update Border Nodes");
    const CSRGraph &graph = _flow_network.graph;

    const std::uint8_t side_status = source_side ? NodeStatus::kSource : NodeStatus::kSink;
    const std::uint8_t other_side_status = source_side ? NodeStatus::kSink : NodeStatus::kSource;

    const NodeStatus &cut_status = _max_flow_algorithm->node_status();

    border_nodes.clear();
    _piercing_heuristic.reset(source_side);
    _piercing_nodes_candidates_marker.reset();

    for (const NodeID u : potential_border_nodes) {
      bool is_border_node = false;

      graph.adjacent_nodes(u, [&](const NodeID v) {
        if (_node_status.has_status(v, side_status) ||
            cut_status.has_status(v, other_side_status)) {
          return;
        }

        is_border_node = true;
        if (!_piercing_nodes_candidates_marker.get(v)) {
          _piercing_nodes_candidates_marker.set(v);

          const bool reachable = _node_status.has_status(v, other_side_status);
          _piercing_heuristic.add_piercing_node_candidate(v, reachable);
        }
      });

      if (is_border_node) {
        border_nodes.push_back(u);
      }
    }
  }

  template <typename BlockFetcher> void compute_moves(BlockFetcher &&fetch_block) {
    SCOPED_TIMER("Compute Moves");

    _constrained_moves.clear();
    for (const auto &[u, u_local] : _flow_network.global_to_local_mapping) {
      const BlockID old_block = _partition[u];
      const BlockID new_block = fetch_block(u_local);

      if (old_block != new_block) {
        _constrained_moves.emplace_back(u, old_block, new_block);
      }
    }
  }

  void initialize_rebalancer() {
    SCOPED_TIMER("Initialize Rebalancer");

    for (const NodeID u : _graph.nodes()) {
      _p_graph_rebalancing_copy.set_block(u, _partition[u]);
    }

    if (_f_ctx.dynamic_rebalancer) {
      _dynamic_balancer.setup(_p_graph_rebalancing_copy, _graph);
    } else {
      _source_side_balancer.setup(
          _block1, _p_graph_rebalancing_copy, _graph, _flow_network.global_to_local_mapping
      );
      _sink_side_balancer.setup(
          _block2, _p_graph_rebalancing_copy, _graph, _flow_network.global_to_local_mapping
      );
    }
  }

  template <typename BlockFetcher>
  void rebalance(const bool source_side, const EdgeWeight cut_value, BlockFetcher &&fetch_block) {
    TIMED_SCOPE("Initialize Partitioned Graph") {
      for (const auto [u, u_local] : _flow_network.global_to_local_mapping) {
        _p_graph_rebalancing_copy.set_block(u, fetch_block(u_local));
      }
    };

    KASSERT(
        metrics::edge_cut_seq(_p_graph_rebalancing_copy) == cut_value,
        "Given an incorrect cut value for partitioned graph",
        assert::heavy
    );

    const BlockID overloaded_block = source_side ? _block1 : _block2;
    const auto [balanced, gain, moved_nodes] = [&] {
      SCOPED_TIMER("Rebalance");

      if (_f_ctx.dynamic_rebalancer) {
        return _dynamic_balancer.rebalance(overloaded_block);
      } else {
        if (source_side) {
          return _source_side_balancer.rebalance();
        } else {
          return _sink_side_balancer.rebalance();
        }
      }
    }();

    if (!balanced) {
      DBG << "Rebalancing failed to produce a balanced cut";
    } else {
      KASSERT(
          metrics::is_balanced(_p_graph_rebalancing_copy, _p_ctx),
          "Rebalancing resulted in an inbalanced partition",
          assert::heavy
      );
      SCOPED_TIMER("Compute Moves");

      const EdgeWeight rebalanced_cut_value = cut_value - gain;
      DBG << "Rebalanced imbalanced cut with resulting global value " << rebalanced_cut_value;

      KASSERT(
          metrics::edge_cut_seq(_p_graph_rebalancing_copy) == rebalanced_cut_value,
          "Given an incorrect cut value for rebalanced partitioned graph",
          assert::heavy
      );

      if (rebalanced_cut_value < _unconstrained_cut_value) {
        _unconstrained_cut_value = rebalanced_cut_value;
        _unconstrained_moves.clear();

        for (const auto [u, _] : _flow_network.global_to_local_mapping) {
          const BlockID old_block = _partition[u];
          const BlockID new_block = _p_graph_rebalancing_copy.block(u);

          if (old_block != new_block) {
            _unconstrained_moves.emplace_back(u, old_block, new_block);
          };
        }

        for (const NodeID u : moved_nodes) {
          if (_flow_network.global_to_local_mapping.contains(u)) {
            continue;
          }

          const BlockID new_block = _p_graph_rebalancing_copy.block(u);
          _unconstrained_moves.emplace_back(u, overloaded_block, new_block);
        }
      }
    }

    TIMED_SCOPE("Reset Partitioned Graph") {
      for (const NodeID u : moved_nodes) {
        _p_graph_rebalancing_copy.set_block(u, overloaded_block);
      }
    };
  }

  [[nodiscard]] bool time_limit_exceeded() const {
    using namespace std::chrono;

    TimePoint current_time = Clock::now();
    std::size_t time_elapsed = duration_cast<milliseconds>(current_time - _start_time).count();

    return time_elapsed >= _f_ctx.time_limit * 60 * 1000;
  }

private:
  const PartitionContext &_p_ctx;
  const TwowayFlowRefinementContext &_f_ctx;

  const PartitionedCSRGraph &_p_graph;
  const CSRGraph &_graph;

  const TimePoint &_start_time;
  bool _time_limit_exceeded;

  BlockID _block1;
  BlockID _block2;

  StaticArray<BlockID> _partition;
  StaticArray<BlockWeight> _block_weights;

  BorderRegion _border_region1;
  BorderRegion _border_region2;
  FlowNetwork _flow_network;

  NodeStatus _node_status;

  ScalableVector<NodeID> _source_side_border_nodes;
  ScalableVector<NodeID> _source_side_border_nodes_candidates;

  ScalableVector<NodeID> _sink_side_border_nodes;
  ScalableVector<NodeID> _sink_side_border_nodes_candidates;

  Marker<> _piercing_nodes_candidates_marker;

  EdgeWeight _global_cut_value;
  EdgeWeight _initial_cut_value;

  EdgeWeight _constrained_cut_value;
  ScalableVector<Move> _constrained_moves;

  EdgeWeight _unconstrained_cut_value;
  ScalableVector<Move> _unconstrained_moves;

  std::unique_ptr<MaxFlowAlgorithm> _max_flow_algorithm;
  PiercingHeuristic _piercing_heuristic;

  PartitionedCSRGraph _p_graph_rebalancing_copy;
  DynamicGreedyBalancer<PartitionedCSRGraph, CSRGraph> _dynamic_balancer;
  StaticGreedyBalancer<PartitionedCSRGraph, CSRGraph> _source_side_balancer;
  StaticGreedyBalancer<PartitionedCSRGraph, CSRGraph> _sink_side_balancer;
};

class SequentialBlockPairScheduler {
  SET_DEBUG(true);

  using Clock = BipartitionFlowRefiner::Clock;
  using TimePoint = BipartitionFlowRefiner::TimePoint;
  using Move = BipartitionFlowRefiner::Move;

public:
  SequentialBlockPairScheduler(const TwowayFlowRefinementContext &f_ctx) : _f_ctx(f_ctx) {}

  bool refine(PartitionedCSRGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx) {
    _p_graph = &p_graph;
    _graph = &graph;

    // Since the timers have a significant running time overhead, we disable them usually.
    IF_NOT_DBG DISABLE_TIMERS();

    activate_all_blocks();

    TimePoint start_time = Clock::now();
    BipartitionFlowRefiner refiner(p_ctx, _f_ctx, p_graph, graph, start_time);

    std::size_t num_round = 0;
    bool found_improvement = false;
    EdgeWeight prev_cut_value = metrics::edge_cut_seq(p_graph);
    while (prev_cut_value > 0) {
      num_round += 1;
      DBG << "Starting round " << num_round;

      EdgeWeight cut_value = prev_cut_value;
      while (!_block_pairs.empty()) {
        const auto [block1, block2] = _block_pairs.front();
        _block_pairs.pop();

        DBG << "Scheduling block pair " << block1 << " and " << block2;
        const auto [time_limit_exceeded, gain, moves] = refiner.refine(cut_value, block1, block2);

        if (time_limit_exceeded) {
          LOG_WARNING << "Time limit exceeded during flow refinement";
          num_round = _f_ctx.max_num_rounds;
          break;
        }

        DBG << "Found balanced cut for bock pair " << block1 << " and " << block2 << " with gain "
            << gain << " (" << cut_value << " -> " << (cut_value - gain) << ")";

        if (gain > 0) {
          apply_moves(moves);

          KASSERT(
              metrics::edge_cut_seq(p_graph) == cut_value - gain,
              "Computed an invalid gain",
              assert::heavy
          );

          KASSERT(
              metrics::is_balanced(p_graph, p_ctx),
              "Computed an imbalanced move sequence",
              assert::heavy
          );

          cut_value -= gain;
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

  void apply_moves(std::span<const Move> moves) {
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
  SET_DEBUG(true);

  using Clock = BipartitionFlowRefiner::Clock;
  using TimePoint = BipartitionFlowRefiner::TimePoint;
  using Move = BipartitionFlowRefiner::Move;

public:
  ParallelBlockPairScheduler(const TwowayFlowRefinementContext &f_ctx) : _f_ctx(f_ctx) {}

  bool refine(PartitionedCSRGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx) {
    _p_graph = &p_graph;
    _graph = &graph;

    // Since timers are not multi-threaded, we disable them during parallel refinement.
    DISABLE_TIMERS();

    activate_all_blocks();

    TimePoint start_time = Clock::now();
    auto refiner_ets = tbb::enumerable_thread_specific<BipartitionFlowRefiner>([&]() {
      return BipartitionFlowRefiner(p_ctx, _f_ctx, p_graph, graph, start_time);
    });

    bool found_improvement = false;
    std::size_t num_round = 0;
    EdgeWeight prev_cut_value = metrics::edge_cut_seq(p_graph);
    while (prev_cut_value > 0) {
      num_round += 1;
      DBG << "Starting round " << num_round;

      EdgeWeight cut_value = prev_cut_value;
      tbb::parallel_for<std::size_t>(0, _block_pairs.size(), [&](const std::size_t i) {
        const auto [block1, block2] = _block_pairs[i];
        auto &refiner = refiner_ets.local();

        DBG << "Scheduling block pair " << block1 << " and " << block2;
        auto [time_limit_exceeded, gain, moves] = refiner.refine(cut_value, block1, block2);

        if (time_limit_exceeded) {
          if (tbb::task::current_context()->cancel_group_execution()) {
            LOG_WARNING << "Time limit exceeded during flow refinement";
            num_round = _f_ctx.max_num_rounds;
          }

          return;
        }

        if (gain <= 0) {
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
        const EdgeWeight actual_gain = cut_value - new_cut_value;
        DBG << "Found balanced cut for bock pair " << block1 << " and " << block2 << " with gain "
            << actual_gain << " (" << cut_value << " -> " << new_cut_value << ")";

        if (actual_gain <= 0) {
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

  void apply_moves(std::span<Move> moves) {
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

  void revert_moves(std::span<const Move> moves) {
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
