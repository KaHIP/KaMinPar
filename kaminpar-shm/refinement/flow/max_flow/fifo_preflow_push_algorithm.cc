#include "kaminpar-shm/refinement/flow/max_flow/fifo_preflow_push_algorithm.h"

#include <algorithm>
#include <queue>
#include <utility>

namespace kaminpar::shm {

FIFOPreflowPushAlgorithm::FIFOPreflowPushAlgorithm(const FIFOPreflowPushContext &ctx) : _ctx(ctx) {}

void FIFOPreflowPushAlgorithm::initialize(
    const CSRGraph &graph,
    std::span<const NodeID> reverse_edges,
    const NodeID source,
    const NodeID sink
) {
  _graph = &graph;
  _reverse_edges = reverse_edges;

  _node_status.initialize(graph.n());
  _node_status.add_source(source);
  _node_status.add_sink(sink);

  _flow_value = 0;

  if (_flow.size() != graph.m()) {
    _flow.resize(graph.m(), static_array::noinit);
  }
  std::fill(_flow.begin(), _flow.end(), 0);

  _grt = GlobalRelabelingThreshold(graph.n(), graph.m(), _ctx.global_relabeling_frequency);

  if (_excess.size() < graph.n()) {
    _excess.resize(graph.n(), static_array::noinit);
  }
  std::fill_n(_excess.begin(), graph.n(), 0);

  if (_cur_edge_offsets.size() < graph.n()) {
    _cur_edge_offsets.resize(graph.n(), static_array::noinit);
  }
  std::fill_n(_cur_edge_offsets.begin(), graph.n(), 0);

  if (_heights.size() < graph.n()) {
    _heights.resize(graph.n(), static_array::noinit);
  }

  saturate_source_edges(_node_status.source_nodes());
  global_relabel();
}

void FIFOPreflowPushAlgorithm::add_sources(std::span<const NodeID> sources) {
  for (const NodeID u : sources) {
    KASSERT(!_node_status.is_sink(u));

    if (_node_status.is_unknown(u)) {
      _node_status.add_source(u);
    }
  }
}

void FIFOPreflowPushAlgorithm::add_sinks(std::span<const NodeID> sinks) {
  for (const NodeID u : sinks) {
    KASSERT(!_node_status.is_source(u));

    if (_node_status.is_unknown(u)) {
      _node_status.add_sink(u);
    }
  }
}

void FIFOPreflowPushAlgorithm::pierce_nodes(std::span<const NodeID> nodes, const bool source_side) {
  if (source_side) {
    add_sources(nodes);

    constexpr bool kSetSourceHeight = true;
    saturate_source_edges<kSetSourceHeight>(nodes);
  } else {
    add_sinks(nodes);

    saturate_source_edges(_node_status.source_nodes());
    global_relabel();
  }
}

MaxFlowAlgorithm::Result FIFOPreflowPushAlgorithm::compute_max_flow() {
  while (!_active_nodes.empty()) {
    const NodeID u = _active_nodes.front();
    _active_nodes.pop();

    discharge(u);

    if (_ctx.global_relabeling_heuristic && _grt.is_reached()) {
      global_relabel();
    }
  }

  IF_DBG debug::print_flow(*_graph, _node_status, _flow);

  KASSERT(
      debug::is_valid_flow(*_graph, _node_status, _flow),
      "computed an invalid flow using preflow-push",
      assert::heavy
  );

  KASSERT(
      debug::is_max_flow(*_graph, _node_status, _flow),
      "computed a non-maximum flow using preflow-push",
      assert::heavy
  );

  KASSERT(
      _flow_value == debug::flow_value(*_graph, _node_status, _flow),
      "computed an invalid flow value using preflow-push",
      assert::heavy
  );

  return Result(_flow_value, _flow);
}

const NodeStatus &FIFOPreflowPushAlgorithm::node_status() const {
  return _node_status;
}

template <bool kSetSourceHeight>
void FIFOPreflowPushAlgorithm::saturate_source_edges(std::span<const NodeID> sources) {
  for (const NodeID source : sources) {
    if constexpr (kSetSourceHeight) {
      _heights[source] = _graph->n();
    }

    _graph->neighbors(source, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
      if (_node_status.is_source(v)) {
        return;
      }

      const EdgeWeight e_flow = _flow[e];
      const EdgeWeight residual_capacity = c - e_flow;

      constexpr bool kFromSource = true;
      push<kFromSource>(source, v, e, residual_capacity);
    });
  }
}

void FIFOPreflowPushAlgorithm::global_relabel() {
  _grt.clear();

  const NodeID num_nodes = _graph->n();
  const NodeID max_level = 2 * num_nodes;
  std::fill_n(_heights.begin(), num_nodes, max_level);

  std::queue<std::pair<NodeID, NodeID>> bfs_queue;
  for (const NodeID terminal : _node_status.sink_nodes()) {
    _heights[terminal] = 0;
    bfs_queue.emplace(terminal, 0);
  }

  while (!bfs_queue.empty()) {
    const auto [u, u_height] = bfs_queue.front();
    bfs_queue.pop();

    const NodeID v_height = u_height + 1;
    _graph->neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
      if (_node_status.is_terminal(v) || _heights[v] != max_level || -_flow[e] == c) {
        return;
      }

      _heights[v] = v_height;
      bfs_queue.emplace(v, v_height);
    });
  }

  for (const NodeID source : _node_status.source_nodes()) {
    _heights[source] = num_nodes;
  }
}

void FIFOPreflowPushAlgorithm::discharge(const NodeID u) {
  const EdgeID first_edge = _graph->first_edge(u);
  const NodeID degree = _graph->degree(u);

  NodeID u_height = _heights[u];
  while (_excess[u] > 0) {
    const EdgeID cur_edge_offset = _cur_edge_offsets[u];

    if (cur_edge_offset == degree) {
      _grt.add_work(degree);

      _cur_edge_offsets[u] = 0;
      u_height = relabel(u);
    } else {
      const EdgeID e = first_edge + cur_edge_offset;
      const NodeID v = _graph->edge_target(e);

      const EdgeWeight e_flow = _flow[e];
      const EdgeWeight e_capacity = _graph->edge_weight(e);

      const EdgeWeight residual_capacity = e_capacity - e_flow;
      if (residual_capacity > 0 && u_height > _heights[v]) {
        push(u, v, e, residual_capacity);
      }

      _cur_edge_offsets[u] += 1;
    }
  }
}

template <bool kFromSource>
void FIFOPreflowPushAlgorithm::push(
    const NodeID from, const NodeID to, const EdgeID e, const EdgeWeight residual_capacity
) {
  const bool from_source = kFromSource ? true : _node_status.is_source(from);
  const EdgeWeight flow =
      from_source ? residual_capacity : std::min(_excess[from], residual_capacity);

  if (flow == 0) {
    return;
  }

  _flow[e] += flow;
  _flow[_reverse_edges[e]] -= flow;

  const EdgeWeight to_prev_excess = _excess[to];
  _excess[to] = to_prev_excess + flow;
  _excess[from] -= flow;

  if (from_source) {
    _flow_value += flow;
  }
  if (_node_status.is_source(to)) {
    _flow_value -= flow;
  }

  const bool to_was_inactive = to_prev_excess == 0;
  if (to_was_inactive && !_node_status.is_terminal(to)) {
    _active_nodes.push(to);
  }
}

NodeID FIFOPreflowPushAlgorithm::relabel(const NodeID u) {
  NodeID min_neighboring_height = std::numeric_limits<NodeID>::max();
  _graph->neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
    const bool has_residual_capacity = _flow[e] < c;
    if (has_residual_capacity) {
      min_neighboring_height = std::min(min_neighboring_height, _heights[v]);
    }
  });

  KASSERT(min_neighboring_height != std::numeric_limits<NodeID>::max());
  const NodeID new_height = min_neighboring_height + 1;

  _heights[u] = new_height;
  return new_height;
}

} // namespace kaminpar::shm
