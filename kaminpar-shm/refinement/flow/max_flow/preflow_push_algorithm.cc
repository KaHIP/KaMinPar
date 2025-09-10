#include "kaminpar-shm/refinement/flow/max_flow/preflow_push_algorithm.h"

#include <algorithm>

namespace kaminpar::shm {

PreflowPushAlgorithm::PreflowPushAlgorithm(const PreflowPushContext &ctx) : _ctx(ctx) {}

void PreflowPushAlgorithm::initialize(
    const CSRGraph &graph,
    std::span<const EdgeID> reverse_edges,
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

  _force_global_relabel = true;
  _grt = GlobalRelabelingThreshold(graph.n(), graph.m(), _ctx.global_relabeling_frequency);

  _nodes_to_desaturate.clear();
  _nodes_to_desaturate.push_back(source);

  if (_cur_edge_offsets.size() < graph.n()) {
    _cur_edge_offsets.resize(graph.n(), static_array::noinit);
  }
  std::fill_n(_cur_edge_offsets.begin(), graph.n(), 0);

  if (_excess.size() < graph.n()) {
    _excess.resize(graph.n(), static_array::noinit);
  }
  std::fill_n(_excess.begin(), graph.n(), 0);

  if (_heights.size() < graph.n()) {
    _heights.resize(graph.n(), static_array::noinit);
  }

  KASSERT(_active_nodes.empty());
}

void PreflowPushAlgorithm::add_sources(std::span<const NodeID> sources) {
  const NodeID num_nodes = _graph->n();

  for (const NodeID u : sources) {
    KASSERT(!_node_status.is_sink(u));

    if (_node_status.is_unknown(u)) {
      _node_status.add_source(u);
      _heights[u] = num_nodes;
    }
  }
}

void PreflowPushAlgorithm::add_sinks(std::span<const NodeID> sinks) {
  for (const NodeID u : sinks) {
    KASSERT(!_node_status.is_source(u));

    if (_node_status.is_unknown(u)) {
      _node_status.add_sink(u);
      _heights[u] = 0;

      _flow_value += _excess[u];
    }
  }
}

void PreflowPushAlgorithm::pierce_nodes(const bool source_side, std::span<const NodeID> nodes) {
  if (source_side) {
    add_sources(nodes);
    for (const NodeID node : nodes) {
      _nodes_to_desaturate.push_back(node);
    }
  } else {
    add_sinks(nodes);
    _force_global_relabel = true;
  }
}

MaxPreflowAlgorithm::Result PreflowPushAlgorithm::compute_max_preflow() {
  IF_STATS _stats.reset();

  saturate_source_edges();
  if (_force_global_relabel) {
    global_relabel<kCollectActiveNodesTag>();
  }

  while (!_active_nodes.empty()) {
    const NodeID u = _active_nodes.front();
    _active_nodes.pop();

    KASSERT(!_node_status.is_terminal(u));
    discharge(u);

    if (_ctx.global_relabeling_heuristic && _grt.is_reached()) {
      global_relabel();
    }
  }

  IF_STATS _stats.print();
  IF_DBG debug::print_flow(*_graph, _node_status, _flow);

  KASSERT(
      debug::is_valid_labeling(*_graph, _node_status, _flow, _heights),
      "computed an invalid labeling using preflow-push",
      assert::heavy
  );

  KASSERT(
      debug::is_valid_preflow(*_graph, _node_status, _flow),
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

  return {_flow_value, _flow};
}

void PreflowPushAlgorithm::saturate_source_edges() {
  for (const NodeID source : _nodes_to_desaturate) {
    KASSERT(_node_status.is_source(source));

    _graph->neighbors(source, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
      if (_node_status.is_source(v)) {
        return;
      }

      const EdgeWeight e_flow = _flow[e];
      const EdgeWeight e_capacity = w;

      const EdgeWeight residual_capacity = e_capacity - e_flow;
      if (residual_capacity <= 0) {
        return;
      }

      const EdgeID e_reverse = _reverse_edges[e];
      _flow[e] += residual_capacity;
      _flow[e_reverse] -= residual_capacity;

      const EdgeWeight to_prev_excess = _excess[v];
      _excess[v] = to_prev_excess + residual_capacity;

      if (_node_status.is_sink(v)) {
        _flow_value += residual_capacity;
      } else if (to_prev_excess == 0) {
        _active_nodes.push(v);
      }
    });
  }

  _nodes_to_desaturate.clear();
}

template <bool kCollectActiveNodes> void PreflowPushAlgorithm::global_relabel() {
  IF_STATS _stats.num_global_relabels += 1;

  _grt.clear();
  _force_global_relabel = false;

  const NodeID num_nodes = _graph->n();
  const NodeID max_level = 2 * num_nodes;
  std::fill_n(_heights.begin(), num_nodes, max_level);

  _bfs_runner.reset();
  for (const NodeID sink : _node_status.sink_nodes()) {
    _heights[sink] = 0;
    _bfs_runner.add_seed(sink);
  }

  _bfs_runner.perform([&](const NodeID u, const NodeID u_height, auto &queue) {
    const NodeID v_height = u_height + 1;

    _graph->neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
      if (_node_status.is_terminal(v) || -_flow[e] == c || _heights[v] != max_level) {
        return;
      }

      _heights[v] = v_height;
      queue.push_back(v);

      if constexpr (kCollectActiveNodes) {
        if (_excess[v] > 0 && v_height < num_nodes) {
          _active_nodes.push(v);
        }
      }
    });
  });

  for (const NodeID source : _node_status.source_nodes()) {
    _heights[source] = num_nodes;
  }

  KASSERT(
      debug::is_valid_labeling(*_graph, _node_status, _flow, _heights),
      "computed an invalid labeling using preflow-push",
      assert::heavy
  );
}

void PreflowPushAlgorithm::discharge(const NodeID u) {
  IF_STATS _stats.num_discharges += 1;

  const EdgeID first_edge = _graph->first_edge(u);
  const NodeID degree = _graph->degree(u);
  const NodeID num_nodes = _graph->n();

  NodeID u_height = _heights[u];
  EdgeID cur_edge_offset = _cur_edge_offsets[u];
  while (_excess[u] > 0 && u_height < num_nodes) {
    if (cur_edge_offset == degree) [[unlikely]] {
      _grt.add_work(degree);

      u_height = relabel(u);
      cur_edge_offset = 0;
      continue;
    }

    const EdgeID e = first_edge + cur_edge_offset++;
    const NodeID v = _graph->edge_target(e);
    if (u_height <= _heights[v]) {
      continue;
    }

    const EdgeWeight e_flow = _flow[e];
    const EdgeWeight e_capacity = _graph->edge_weight(e);

    const EdgeWeight residual_capacity = e_capacity - e_flow;
    const EdgeWeight flow = std::min(_excess[u], residual_capacity);
    if (flow <= 0) {
      continue;
    }

    const EdgeID e_reverse = _reverse_edges[e];
    _flow[e] += flow;
    _flow[e_reverse] -= flow;

    const EdgeWeight to_prev_excess = _excess[v];
    _excess[v] = to_prev_excess + flow;
    _excess[u] -= flow;

    if (_node_status.is_sink(v)) {
      _flow_value += flow;
      continue;
    }

    const bool to_was_inactive = to_prev_excess == 0;
    if (to_was_inactive && !_node_status.is_terminal(v)) {
      _active_nodes.push(v);
    }
  }

  _cur_edge_offsets[u] = cur_edge_offset;
  _heights[u] = u_height;
}

NodeID PreflowPushAlgorithm::relabel(const NodeID u) {
  NodeID min_neighboring_height = std::numeric_limits<NodeID>::max();
  _graph->neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
    const bool has_residual_capacity = _flow[e] < c;
    if (has_residual_capacity) {
      min_neighboring_height = std::min(min_neighboring_height, _heights[v]);
    }
  });

  KASSERT(min_neighboring_height != std::numeric_limits<NodeID>::max());
  return min_neighboring_height + 1;
}

std::span<const NodeID> PreflowPushAlgorithm::excess_nodes() {
  _excess_nodes.clear();

  for (const NodeID u : _graph->nodes()) {
    if (_excess[u] > 0 && !_node_status.is_terminal(u)) {
      _excess_nodes.push_back(u);
    }
  }

  return _excess_nodes;
}

const NodeStatus &PreflowPushAlgorithm::node_status() const {
  return _node_status;
}

void PreflowPushAlgorithm::free() {
  _node_status.free();

  _excess_nodes.clear();
  _excess_nodes.shrink_to_fit();

  _flow.free();

  _bfs_runner.free();

  _nodes_to_desaturate.clear();
  _nodes_to_desaturate.shrink_to_fit();

  _cur_edge_offsets.free();
  _excess.free();
  _heights.free();
  _active_nodes = {};
}

} // namespace kaminpar::shm
