#include "kaminpar-shm/refinement/flow/multiway_cut/labelling_function_heuristic.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

#include "kaminpar-shm/refinement/flow/max_flow/edmond_karp_algorithm.h"
#include "kaminpar-shm/refinement/flow/max_flow/fifo_preflow_push_algorithm.h"
#include "kaminpar-shm/refinement/flow/util/reverse_edge_index.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

LabellingFunctionHeuristic::LabellingFunctionHeuristic(const LabellingFunctionHeuristicContext &ctx)
    : _ctx(ctx) {
  switch (FlowAlgorithm::FIFO_PREFLOW_PUSH) {
  case FlowAlgorithm::EDMONDS_KARP:
    _max_flow_algorithm = std::make_unique<EdmondsKarpAlgorithm>();
    break;
  case FlowAlgorithm::FIFO_PREFLOW_PUSH:
    _max_flow_algorithm = std::make_unique<FIFOPreflowPushAlgorithm>(ctx.fifo_preflow_push);
    break;
  }
}

MultiwayCutAlgorithm::Result LabellingFunctionHeuristic::compute(
    const CSRGraph &graph, const std::vector<std::unordered_set<NodeID>> &terminal_sets
) {
  _graph = &graph;
  _reverse_edge_index = compute_reverse_edge_index(graph);

  _terminals.clear();
  for (const auto &terminals : terminal_sets) {
    const NodeID terminal = *terminals.begin();
    _terminals.insert(terminal);
  }

  initialize_labelling_function();

  improve_labelling_function();
  std::unordered_set<EdgeID> cut_edges = derive_cut_edges();

  KASSERT(
      debug::is_valid_multiway_cut(graph, terminal_sets, cut_edges),
      "computed a non-valid multi-way cut using the isolating-cut heuristic",
      assert::heavy
  );

  return Result(_labelling_function_cost, std::move(cut_edges));
}

MultiwayCutAlgorithm::Result LabellingFunctionHeuristic::compute(
    const PartitionedCSRGraph &p_graph,
    const CSRGraph &graph,
    const std::vector<std::unordered_set<NodeID>> &terminal_sets
) {
  _graph = &graph;
  _reverse_edge_index = compute_reverse_edge_index(graph);

  _terminals.clear();
  for (const auto &terminals : terminal_sets) {
    const NodeID terminal = *terminals.begin();
    _terminals.insert(terminal);
  }

  if (_ctx.initialization_strategy == LabellingFunctionInitializationStrategy::EXISTING_PARTITION) {
    initialize_labelling_function(p_graph);
  } else {
    initialize_labelling_function();
  }

  improve_labelling_function();
  std::unordered_set<EdgeID> cut_edges = derive_cut_edges();

  KASSERT(
      debug::is_valid_multiway_cut(graph, terminal_sets, cut_edges),
      "computed a non-valid multi-way cut using the isolating-cut heuristic",
      assert::heavy
  );

  return Result(_labelling_function_cost, std::move(cut_edges));
}

void LabellingFunctionHeuristic::initialize_labelling_function() {
  SCOPED_TIMER("Initialize Labelling Function");

  BlockID label = 0;
  _terminal_labels.clear();
  for (const NodeID terminal : _terminals) {
    _terminal_labels[terminal] = label++;
  }

  if (_labelling_function.size() < _graph->n()) {
    _labelling_function.resize(_graph->n(), static_array::noinit);
  }

  const BlockID num_terminals = _terminals.size();
  const bool initialize_with_random_values =
      _ctx.initialization_strategy == LabellingFunctionInitializationStrategy::RANDOM;
  Random &random = Random::instance();

  for (const NodeID u : _graph->nodes()) {
    if (_terminals.contains(u)) {
      _labelling_function[u] = _terminal_labels[u];
    } else if (initialize_with_random_values) {
      _labelling_function[u] = random.random_index(0, num_terminals);
    } else {
      _labelling_function[u] = 0;
    }
  }
}

void LabellingFunctionHeuristic::initialize_labelling_function(const PartitionedCSRGraph &p_graph) {
  SCOPED_TIMER("Initialize Labelling Function");

  BlockID label = 0;
  _terminal_labels.clear();
  for (const NodeID terminal : _terminals) {
    _terminal_labels[terminal] = label++;
  }

  if (_labelling_function.size() < _graph->n()) {
    _labelling_function.resize(_graph->n(), static_array::noinit);
  }

  for (const NodeID u : _graph->nodes()) {
    if (_terminals.contains(u)) {
      _labelling_function[u] = _terminal_labels[u];
    } else {
      _labelling_function[u] = p_graph.block(u);
    }
  }
}

void LabellingFunctionHeuristic::improve_labelling_function() {
  KASSERT(is_valid_labelling_function(), "invalid labelling function");

  EdgeWeight cur_cost = compute_labelling_function_cost();
  DBG << "Starting labelling function local search with a initial cost " << cur_cost;

  const BlockID num_terminals = _terminals.size();
  while (true) {
    bool found_improvement = false;

    for (BlockID terminal = 0; terminal < num_terminals; ++terminal) {
      FlowNetwork flow_network = construct_flow_network(terminal);

      DBG << "Constructed a flow network with n=" << flow_network.graph.n()
          << " and m=" << flow_network.graph.m();

      TIMED_SCOPE("Initialize Max Flow Algorithm") {
        _max_flow_algorithm->initialize(
            flow_network.graph, _reverse_edge_index, flow_network.source, flow_network.sink
        );
      };

      const auto [cost, flow] = TIMED_SCOPE("Compute Max Flow") {
        return _max_flow_algorithm->compute_max_flow();
      };
      DBG << "Computed a labelling function with cost " << cost;

      if (cost < (1 - _ctx.epsilon) * cur_cost) {
        found_improvement = true;
        cur_cost = cost;

        derive_labelling_function(terminal, flow_network, flow);

        KASSERT(is_valid_labelling_function(), "invalid labelling function", assert::heavy);
        KASSERT(
            cost == compute_labelling_function_cost(),
            "invalid labelling function cost",
            assert::heavy
        );
      }
    }

    if (!found_improvement) {
      break;
    }
  }

  _labelling_function_cost = cur_cost;
}

EdgeWeight LabellingFunctionHeuristic::compute_labelling_function_cost() const {
  SCOPED_TIMER("Compute Labelling Function Cost");

  EdgeWeight cost = 0;

  for (const NodeID u : _graph->nodes()) {
    _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      cost += (_labelling_function[u] != _labelling_function[v]) ? w : 0;
    });
  }

  return cost / 2;
}

LabellingFunctionHeuristic::FlowNetwork
LabellingFunctionHeuristic::construct_flow_network(const BlockID label) const {
  SCOPED_TIMER("Construct Flow Network");

  constexpr NodeID kSource = 0;
  constexpr NodeID kSink = 1;
  constexpr NodeID kFirstNodeID = 2;
  constexpr EdgeWeight kInfinity = std::numeric_limits<EdgeWeight>::max();
  const NodeID num_terminals = _terminals.size();

  NodeID num_nodes = 2 + _graph->n();
  for (const NodeID u : _graph->nodes()) {
    const BlockID u_label = _labelling_function[u];

    _graph->adjacent_nodes(u, [&](const NodeID v) {
      if (u >= v) {
        return;
      }

      const BlockID v_label = _labelling_function[v];
      num_nodes += (u_label != v_label) ? 1 : 0;
    });
  }

  StaticArray<EdgeID> nodes(num_nodes + 1, static_array::noinit);
  std::fill_n(nodes.begin(), nodes.size(), 0);

  nodes[kSource] = num_terminals - 1;
  for (const NodeID terminal : _terminals) {
    nodes[kFirstNodeID + terminal] += (_labelling_function[terminal] != label) ? 1 : 0;
  }

  NodeID cur_edge = kFirstNodeID + _graph->n();
  for (const NodeID u : _graph->nodes()) {
    const BlockID u_label = _labelling_function[u];
    if (u_label == label) {
      nodes[kSink] += 1;
      nodes[kFirstNodeID + u] += 1;
    }

    _graph->adjacent_nodes(u, [&](const NodeID v) {
      if (u >= v) {
        return;
      }

      const BlockID v_label = _labelling_function[v];
      if (u_label == v_label) {
        nodes[kFirstNodeID + u] += 1;
        nodes[kFirstNodeID + v] += 1;
      } else {
        nodes[kSink] += 1;
        nodes[cur_edge] += 1;

        if (u_label != label) {
          nodes[kFirstNodeID + u] += 1;
          nodes[cur_edge] += 1;
        }

        if (v_label != label) {
          nodes[kFirstNodeID + v] += 1;
          nodes[cur_edge] += 1;
        }

        cur_edge += 1;
      }
    });
  }
  KASSERT(cur_edge == num_nodes);

  std::partial_sum(nodes.begin(), nodes.end(), nodes.begin());

  const EdgeID num_edges = nodes.back();
  StaticArray<NodeID> edges(num_edges, kInvalidNodeID, static_array::seq);
  StaticArray<EdgeWeight> edge_weights(num_edges, static_array::noinit);

  const auto add_edge = [&](const NodeID u, const NodeID v, const EdgeWeight w) {
    const EdgeID e1 = --nodes[u];
    edges[e1] = v;
    edge_weights[e1] = w;

    const EdgeID e2 = --nodes[v];
    edges[e2] = u;
    edge_weights[e2] = w;

    KASSERT(u != kSource || v != kSink);
  };

  for (const auto &[terminal, terminal_label] : _terminal_labels) {
    if (terminal_label != label) {
      add_edge(kSource, kFirstNodeID + terminal, kInfinity);
    }
  }

  cur_edge = kFirstNodeID + _graph->n();
  for (const NodeID u : _graph->nodes()) {
    const BlockID u_label = _labelling_function[u];
    if (u_label == label) {
      add_edge(kSink, kFirstNodeID + u, kInfinity);
    }

    _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      if (u >= v) {
        return;
      }

      const BlockID v_label = _labelling_function[v];
      if (u_label == v_label) {
        add_edge(kFirstNodeID + u, kFirstNodeID + v, w);
      } else {
        add_edge(kSink, cur_edge, w);

        if (u_label != label) {
          add_edge(kFirstNodeID + u, cur_edge, w);
        }

        if (v_label != label) {
          add_edge(kFirstNodeID + v, cur_edge, w);
        }

        cur_edge += 1;
      }
    });
  }
  KASSERT(cur_edge == num_nodes);

  CSRGraph graph(
      CSRGraph::seq(),
      std::move(nodes),
      std::move(edges),
      StaticArray<NodeWeight>(),
      std::move(edge_weights)
  );
  KASSERT(debug::validate_graph(graph), "constructed invalid flow network", assert::heavy);

  return FlowNetwork(kSource, kSink, std::move(graph));
}

void LabellingFunctionHeuristic::derive_labelling_function(
    const BlockID label, const FlowNetwork &flow_network, std::span<const EdgeWeight> flow
) {
  const std::unordered_set<NodeID> cut_nodes =
      compute_cut_nodes(flow_network.graph, flow_network.source, flow);

  constexpr NodeID kFirstNodeID = 2;
  for (const NodeID u : _graph->nodes()) {
    const NodeID u_local = kFirstNodeID + u;

    if (!cut_nodes.contains(u_local)) {
      _labelling_function[u] = label;
    }
  }
}

std::unordered_set<EdgeID> LabellingFunctionHeuristic::derive_cut_edges() const {
  SCOPED_TIMER("Derive Cut Edges");

  std::unordered_set<EdgeID> cut_edges;

  for (const NodeID u : _graph->nodes()) {
    _graph->neighbors(u, [&](const EdgeID e, const NodeID v) {
      if (_labelling_function[u] != _labelling_function[v]) {
        cut_edges.insert(e);
      }
    });
  }

  return cut_edges;
}

std::unordered_set<NodeID> LabellingFunctionHeuristic::compute_cut_nodes(
    const CSRGraph &graph, const NodeID terminal, std::span<const EdgeWeight> flow
) {
  SCOPED_TIMER("Compute Cut Nodes");

  std::unordered_set<NodeID> cut_nodes;

  std::queue<NodeID> bfs_queue;
  cut_nodes.insert(terminal);
  bfs_queue.push(terminal);

  while (!bfs_queue.empty()) {
    const NodeID u = bfs_queue.front();
    bfs_queue.pop();

    graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
      if (cut_nodes.contains(v)) {
        return;
      }

      const EdgeWeight e_flow = flow[e];
      const bool has_residual_capacity = e_flow < c;
      if (has_residual_capacity) {
        cut_nodes.insert(v);
        bfs_queue.push(v);
      }
    });
  }

  return cut_nodes;
}

bool LabellingFunctionHeuristic::is_valid_labelling_function() const {
  bool is_valid = true;

  for (const auto &[terminal, terminal_label] : _terminal_labels) {
    if (terminal_label != _labelling_function[terminal]) {
      LOG_WARNING << "Terminal " << terminal << " is assigned " << _labelling_function[terminal]
                  << " != " << terminal_label;
      is_valid = false;
    }
  }

  return is_valid;
}

} // namespace kaminpar::shm
