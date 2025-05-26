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
#include <utility>
#include <vector>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#ifdef KAMINPAR_WHFC_FOUND
#include "algorithm/hyperflowcutter.h"
#include "algorithm/parallel_push_relabel.h"
#include "datastructure/flow_hypergraph_builder.h"
#include "definitions.h"
#endif

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/balancer/sequential_greedy_balancer.h"
#include "kaminpar-shm/refinement/flow/max_flow/edmond_karp_algorithm.h"
#include "kaminpar-shm/refinement/flow/max_flow/fifo_preflow_push_algorithm.h"
#include "kaminpar-shm/refinement/flow/max_flow/max_flow_algorithm.h"
#include "kaminpar-shm/refinement/flow/piercing/piercing_heuristic.h"
#include "kaminpar-shm/refinement/flow/util/border_region.h"
#include "kaminpar-shm/refinement/flow/util/node_status.h"
#include "kaminpar-shm/refinement/flow/util/reverse_edge_index.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

class BipartitionFlowRefiner {
  SET_DEBUG(false);

  static constexpr std::uint8_t kStatusUnknown = 0;
  static constexpr std::uint8_t kStatusSourceReachable = 1;
  static constexpr std::uint8_t kStatusSinkReachable = 2;

  struct FlowNetwork {
    NodeID source;
    NodeID sink;

    CSRGraph graph;
    StaticArray<NodeID> reverse_edges;
    std::unordered_map<NodeID, NodeID> global_to_local_mapping;
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
        _graph(graph),
        _partition(graph.n(), static_array::noinit) {
    switch (_f_ctx.flow_algorithm) {
    case FlowAlgorithm::EDMONDS_KARP:
      _max_flow_algorithm = std::make_unique<EdmondsKarpAlgorithm>();
      break;
    case FlowAlgorithm::FIFO_PREFLOW_PUSH:
      _max_flow_algorithm = std::make_unique<FIFOPreflowPushAlgorithm>(_f_ctx.fifo_preflow_push);
      break;
    }
  }

  Result refine(const EdgeWeight cut_value, const BlockID block1, const BlockID block2) {
    KASSERT(block1 != block2, "Only different block pairs can be refined");

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

    if (_f_ctx.use_whfc) {
      run_hyper_flow_cutter();
    } else {
      run_flow_cutter();
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

    BlockWeight block1_weight = 0;
    BlockWeight block2_weight = 0;
    for (const NodeID u : _graph.nodes()) {
      const BlockID u_block = _p_graph.block(u);
      _partition[u] = u_block;

      const NodeWeight u_weight = _graph.node_weight(u);
      block1_weight += (u_block == _block1) ? u_weight : 0;
      block2_weight += (u_block == _block2) ? u_weight : 0;
    }

    _block1_weight = block1_weight;
    _block2_weight = block2_weight;
  }

  void compute_border_regions() {
    SCOPED_TIMER("Compute Border Regions");

    const BlockID block1 = _block1;
    const BlockID block2 = _block2;

    const NodeWeight max_border_region_weight1 =
        (1 + _f_ctx.border_region_scaling_factor * _p_ctx.epsilon()) *
            _p_ctx.perfectly_balanced_block_weight(block2) -
        _block2_weight;
    _border_region1.reset(block1, max_border_region_weight1);

    const NodeWeight max_border_region_weight2 =
        (1 + _f_ctx.border_region_scaling_factor * _p_ctx.epsilon()) *
            _p_ctx.perfectly_balanced_block_weight(block1) -
        _block1_weight;
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

      _graph.adjacent_nodes(u, [&](const NodeID v) {
        if (_partition[v] != block || border_region.contains(v)) {
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
    node_weights[kSource] = _block1_weight - _border_region1.weight();

    nodes[kSink] = num_sink_edges;
    node_weights[kSink] = _block2_weight - _border_region2.weight();

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

    START_TIMER("Run HyperFlowCutter");
    whfc::HyperFlowCutter<whfc::ParallelPushRelabel> flow_cutter(hypergraph, 1);
    flow_cutter.forceSequential(true);
    flow_cutter.setBulkPiercing(false);

    flow_cutter.setFlowBound(_initial_cut_value);
    flow_cutter.cs.setMaxBlockWeight(0, _p_ctx.max_block_weight(_block1));
    flow_cutter.cs.setMaxBlockWeight(1, _p_ctx.max_block_weight(_block2));

    const auto on_cut = [&]() {
      if (_f_ctx.unconstrained && !flow_cutter.cs.isBalanced()) {
        const EdgeWeight gain = _initial_cut_value - flow_cutter.cs.flow_algo.flow_value;
        const EdgeWeight cut_value = _global_cut_value - gain;

        const auto fetch_block = [&](const NodeID u) {
          const bool is_on_source_side = flow_cutter.cs.flow_algo.isSourceReachable(whfc::Node(u));
          return is_on_source_side ? _block1 : _block2;
        };

        PartitionedCSRGraph p_graph = copy_partitioned_graph(fetch_block);
        rebalance(cut_value, p_graph);
      }

      return true;
    };

    const bool success = flow_cutter.enumerateCutsUntilBalancedOrFlowBoundExceeded(
        whfc::Node(_flow_network.source), whfc::Node(_flow_network.sink), on_cut
    );
    STOP_TIMER();

    if (!success) {
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

    const NodeWeight total_weight = _block1_weight + _block2_weight;
    const NodeWeight max_block1_weight = _p_ctx.max_block_weight(_block1);
    const NodeWeight max_block2_weight = _p_ctx.max_block_weight(_block2);

    _node_status.initialize(_flow_network.graph.n());

    PiercingHeuristic piercing_heuristic(
        _f_ctx.piercing, _flow_network.graph, _border_region1.nodes(), _border_region2.nodes()
    );

    TIMED_SCOPE("Initialize Max Flow Algorithm") {
      _max_flow_algorithm->initialize(
          _flow_network.graph, _flow_network.reverse_edges, _flow_network.source, _flow_network.sink
      );
    };

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

      const auto [source_cut_weight, sink_cut_weight] =
          compute_cuts(_max_flow_algorithm->node_status(), flow);

      const bool is_source_cut_balanced = source_cut_weight <= max_block1_weight &&
                                          (total_weight - source_cut_weight) <= max_block2_weight;
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

      const bool is_sink_cut_balanced = sink_cut_weight <= max_block2_weight &&
                                        (total_weight - sink_cut_weight) <= max_block1_weight;
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
        TIMED_SCOPE("Rebalance source-side cut") {
          const EdgeWeight gain = _initial_cut_value - cut_value;
          const EdgeWeight cut_value = _global_cut_value - gain;

          const auto fetch_block = [&](const NodeID u) {
            return _node_status.is_source(u) ? _block1 : _block2;
          };

          PartitionedCSRGraph source_cut_induced_p_graph = copy_partitioned_graph(fetch_block);
          rebalance(cut_value, source_cut_induced_p_graph);
        };

        TIMED_SCOPE("Rebalance sink-side cut") {
          const EdgeWeight gain = _initial_cut_value - cut_value;
          const EdgeWeight cut_value = _global_cut_value - gain;

          const auto fetch_block = [&](const NodeID u) {
            return _node_status.is_sink(u) ? _block2 : _block1;
          };

          PartitionedCSRGraph sink_cut_induced_p_graph = copy_partitioned_graph(fetch_block);
          rebalance(cut_value, sink_cut_induced_p_graph);
        };
      }

      SCOPED_TIMER("Compute Piercing Node");
      if (source_cut_weight <= sink_cut_weight) {
        DBG << "Piercing on source-side (" << source_cut_weight << "/" << max_block1_weight << ", "
            << (total_weight - source_cut_weight) << "/" << max_block2_weight << ")";

        const NodeWeight max_piercing_node_weight = max_block1_weight - source_cut_weight;
        const auto piercing_nodes = piercing_heuristic.pierce_on_source_side(
            _node_status, _max_flow_algorithm->node_status(), max_piercing_node_weight
        );

        if (piercing_nodes.empty()) {
          LOG_WARNING << "Failed to find a suitable piercing node; "
                         "aborting refinement for block pair "
                      << _block1 << " and " << _block2;
          break;
        }

        _max_flow_algorithm->add_sources(_node_status.source_nodes());
        _max_flow_algorithm->pierce_nodes(piercing_nodes, true);
      } else {
        DBG << "Piercing on sink-side (" << sink_cut_weight << "/" << max_block2_weight << ", "
            << (total_weight - sink_cut_weight) << "/" << max_block1_weight << ")";

        const NodeWeight max_piercing_node_weight = max_block2_weight - sink_cut_weight;
        const auto piercing_nodes = piercing_heuristic.pierce_on_sink_side(
            _node_status, _max_flow_algorithm->node_status(), max_piercing_node_weight
        );

        if (piercing_nodes.empty()) {
          LOG_WARNING << "Failed to find a suitable piercing node; "
                         "aborting refinement for block pair "
                      << _block1 << " and " << _block2;
          break;
        }

        _max_flow_algorithm->add_sinks(_node_status.sink_nodes());
        _max_flow_algorithm->pierce_nodes(piercing_nodes, false);
      }
    }
  }

  std::pair<NodeWeight, NodeWeight>
  compute_cuts(const NodeStatus &cut_status, std::span<const EdgeWeight> flow) {
    _node_status.reset();

    const EdgeWeight source_cut_weight = compute_cut(cut_status, flow, true);
    const EdgeWeight sink_cut_weight = compute_cut(cut_status, flow, false);

    return {source_cut_weight, sink_cut_weight};
  }

  NodeWeight compute_cut(
      const NodeStatus &cut_status, std::span<const EdgeWeight> flow, const bool source_side
  ) {
    SCOPED_TIMER("Compute Reachable Nodes");

    NodeWeight cut_weight = 0;

    const CSRGraph &graph = _flow_network.graph;
    std::queue<NodeID> bfs_queue;

    std::span<const NodeID> terminals =
        source_side ? cut_status.source_nodes() : cut_status.sink_nodes();
    for (const NodeID terminal : terminals) {
      bfs_queue.push(terminal);

      cut_weight += graph.node_weight(terminal);
      if (source_side) {
        _node_status.add_source(terminal);
      } else {
        _node_status.add_sink(terminal);
      }
    }

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

          cut_weight += graph.node_weight(v);
          if (source_side) {
            _node_status.add_source(v);
          } else {
            _node_status.add_sink(v);
          }
        }
      });
    }

    return cut_weight;
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

  template <typename BlockFetcher>
  PartitionedCSRGraph copy_partitioned_graph(BlockFetcher &&fetch_block) const {
    SCOPED_TIMER("Copy Partitioned Graph");

    PartitionedCSRGraph p_graph(PartitionedCSRGraph::seq(), _graph, _p_graph.k());

    const std::unordered_map<NodeID, NodeID> &mapping = _flow_network.global_to_local_mapping;
    for (const NodeID u : _graph.nodes()) {
      if (auto it = mapping.find(u); it != mapping.end()) {
        const NodeID u_local = it->second;
        const BlockID u_block = fetch_block(u_local);

        p_graph.set_block(u, u_block);
        continue;
      }

      const BlockID u_block = _partition[u];
      p_graph.set_block(u, u_block);
    }

    return p_graph;
  }

  void rebalance(EdgeWeight cut_value, PartitionedCSRGraph &p_graph) {
    SCOPED_TIMER("Rebalancing");

    KASSERT(
        metrics::edge_cut_seq(p_graph) == cut_value,
        "Given an incorrect cut value for partitioned graph",
        assert::heavy
    );

    const auto [rebalanced, gain] = _balancer.balance(p_graph, _graph, _p_ctx.max_block_weights());
    if (!rebalanced) {
      return;
    }

    const EdgeWeight rebalanced_cut_value = cut_value - gain;
    DBG << "Rebalanced imbalanced cut with resulting global value " << rebalanced_cut_value;

    if (rebalanced_cut_value < _unconstrained_cut_value) {
      _unconstrained_cut_value = rebalanced_cut_value;
      _unconstrained_moves.clear();

      for (const NodeID u : _graph.nodes()) {
        const BlockID old_block = _partition[u];
        const BlockID new_block = p_graph.block(u);

        if (old_block != new_block) {
          _unconstrained_moves.emplace_back(u, old_block, new_block);
        }
      }
    }
  }

private:
  const PartitionContext &_p_ctx;
  const TwowayFlowRefinementContext &_f_ctx;

  const PartitionedCSRGraph &_p_graph;
  const CSRGraph &_graph;

  std::unique_ptr<MaxFlowAlgorithm> _max_flow_algorithm;
  SequentialGreedyBalancerImpl<PartitionedCSRGraph, CSRGraph> _balancer;

  StaticArray<BlockID> _partition;
  NodeStatus _node_status;

  BlockID _block1;
  BlockID _block2;

  BlockWeight _block1_weight;
  BlockWeight _block2_weight;

  BorderRegion _border_region1;
  BorderRegion _border_region2;
  FlowNetwork _flow_network;

  EdgeWeight _global_cut_value;
  EdgeWeight _initial_cut_value;

  EdgeWeight _constrained_cut_value;
  std::vector<Move> _constrained_moves;

  EdgeWeight _unconstrained_cut_value;
  std::vector<Move> _unconstrained_moves;
};

class SequentialBlockPairScheduler {
  SET_DEBUG(true);

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
    while (prev_cut_value > 0) {
      num_round += 1;
      DBG << "Starting round " << num_round;

      EdgeWeight cut_value = prev_cut_value;
      while (!_block_pairs.empty()) {
        const auto [block1, block2] = _block_pairs.front();
        _block_pairs.pop();

        DBG << "Scheduling block pair " << block1 << " and " << block2;
        const auto [gain, moves] = refiner.refine(cut_value, block1, block2);

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
  SET_DEBUG(true);

  using Move = BipartitionFlowRefiner::Move;

public:
  ParallelBlockPairScheduler(const TwowayFlowRefinementContext &f_ctx) : _f_ctx(f_ctx) {}

  bool refine(PartitionedCSRGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx) {
    _p_graph = &p_graph;
    _graph = &graph;

    // Since timers are not multi-threaded, we disable them during parallel refinement.
    DISABLE_TIMERS();

    activate_all_blocks();

    auto refiner_ets = tbb::enumerable_thread_specific<BipartitionFlowRefiner>([&]() {
      return BipartitionFlowRefiner(p_ctx, _f_ctx, p_graph, graph);
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
        auto [expected_gain, moves] = refiner.refine(cut_value, block1, block2);

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
