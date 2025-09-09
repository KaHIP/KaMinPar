#include "kaminpar-shm/refinement/flow/max_flow/parallel_blocked_preflow_push_algorithm.h"

#include <algorithm>
#include <functional>
#include <utility>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "kaminpar-common/assert.h"

namespace kaminpar::shm {

ParallelBlockedPreflowPushAlgorithm::ParallelBlockedPreflowPushAlgorithm(
    const PreflowPushContext &ctx
)
    : _ctx(ctx) {}

void ParallelBlockedPreflowPushAlgorithm::initialize(
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

  _round = 0;

  _force_global_relabel = true;
  _grt = GlobalRelabelingThreshold(graph.n(), graph.m(), _ctx.global_relabeling_frequency);

  _nodes_to_desaturate.clear();
  _nodes_to_desaturate.push_back(source);

  if (_last_activated.size() < graph.n()) {
    _last_activated.resize(graph.n(), static_array::noinit);
  }
  std::fill_n(_last_activated.begin(), graph.n(), 0);

  if (_cur_edge_offsets.size() < graph.n()) {
    _cur_edge_offsets.resize(graph.n(), static_array::noinit);
  }
  std::fill_n(_cur_edge_offsets.begin(), graph.n(), 0);

  if (_active_node_state.size() < graph.n()) {
    _active_node_state.resize(graph.n(), static_array::noinit);
  }
  std::fill_n(_active_node_state.begin(), graph.n(), kNotModifiedState);

  if (_excess.size() < graph.n()) {
    _excess.resize(graph.n(), static_array::noinit);
    _excess_delta.resize(graph.n(), static_array::noinit);
  }
  std::fill_n(_excess.begin(), graph.n(), 0);
  std::fill_n(_excess_delta.begin(), graph.n(), 0);

  if (_heights.size() < graph.n()) {
    _heights.resize(graph.n(), static_array::noinit);
    _next_heights.resize(graph.n(), static_array::noinit);
  }

  _active_nodes.reserve(graph.n());
  _next_active_nodes.reserve(graph.n());
  _next_active_nodes.clear();
}

void ParallelBlockedPreflowPushAlgorithm::add_sources(std::span<const NodeID> sources) {
  const NodeID num_nodes = _graph->n();

  for (const NodeID u : sources) {
    KASSERT(!_node_status.is_sink(u));

    if (_node_status.is_unknown(u)) {
      _node_status.add_source(u);
      _heights[u] = num_nodes;
    }
  }
}

void ParallelBlockedPreflowPushAlgorithm::add_sinks(std::span<const NodeID> sinks) {
  for (const NodeID u : sinks) {
    KASSERT(!_node_status.is_source(u));

    if (_node_status.is_unknown(u)) {
      _node_status.add_sink(u);
      _heights[u] = 0;

      _flow_value += _excess[u];
    }
  }
}

void ParallelBlockedPreflowPushAlgorithm::pierce_nodes(
    const bool source_side, std::span<const NodeID> nodes
) {
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

MaxPreflowAlgorithm::Result ParallelBlockedPreflowPushAlgorithm::compute_max_preflow() {
  IF_STATS _stats.reset();

  _round += 1;
  saturate_source_edges();
  if (_force_global_relabel) {
    global_relabel<kCollectActiveNodesTag>();
  }

  while (!_next_active_nodes.empty()) {
    std::swap(_active_nodes, _next_active_nodes);
    _next_active_nodes.clear();
    _round += 1;

    IF_STATS _stats.num_sequential_rounds +=
        _active_nodes.size() <= _ctx.sequential_discharge_threshold ? 1 : 0;
    while (!_active_nodes.empty() && _active_nodes.size() <= _ctx.sequential_discharge_threshold) {
      const NodeID u = _active_nodes.pop_back();
      _last_activated[u] = _round - 1;

      KASSERT(!_node_status.is_terminal(u));
      discharge(u);

      if (_ctx.global_relabeling_heuristic && _grt.is_reached()) {
        global_relabel();
      }
    }

    if (!_active_nodes.empty()) {
      IF_STATS _stats.num_parallel_rounds += 1;

      discharge_active_nodes();
      apply_updates();

      _grt.add_work(_work_ets.combine(std::plus<>()));
      _work_ets.clear();

      if (_ctx.global_relabeling_heuristic && _grt.is_reached()) {
        global_relabel();
      }
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

  return Result(_flow_value, _flow);
}

void ParallelBlockedPreflowPushAlgorithm::saturate_source_edges() {
  if (_nodes_to_desaturate.empty()) {
    return;
  }

  const NodeID num_nodes = _nodes_to_desaturate.size();
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, num_nodes), [&](const auto &range) {
    BufferedVector<NodeID>::Buffer next_active_nodes = _next_active_nodes.local_buffer();

    for (NodeID i = range.begin(), end = range.end(); i < end; ++i) {
      const NodeID source = _nodes_to_desaturate[i];
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

        if (_node_status.is_sink(v)) {
          __atomic_fetch_add(&_flow_value, residual_capacity, __ATOMIC_RELAXED);
          return;
        }

        __atomic_fetch_add(&_excess[v], residual_capacity, __ATOMIC_RELAXED);
        if (__atomic_exchange_n(&_last_activated[v], _round, __ATOMIC_ACQ_REL) != _round) {
          next_active_nodes.push_back(v);
        }
      });
    }
  });
  _next_active_nodes.flush();

  _nodes_to_desaturate.clear();
}

template <bool kCollectActiveNodes> void ParallelBlockedPreflowPushAlgorithm::global_relabel() {
  IF_STATS _stats.num_global_relabels += 1;

  _grt.clear();
  _force_global_relabel = false;

  const NodeID num_nodes = _graph->n();
  const NodeID max_level = 2 * num_nodes;
  std::fill_n(_heights.begin(), num_nodes, max_level);

  _parallel_bfs_runner.reset(num_nodes);
  for (const NodeID sink : _node_status.sink_nodes()) {
    _heights[sink] = 0;
    _parallel_bfs_runner.add_seed(sink);
  }

  _parallel_bfs_runner.perform([&](const NodeID u, const NodeID u_height, auto queue) {
    if constexpr (kCollectActiveNodes) {
      if (!_node_status.is_terminal(u) && _excess[u] > 0 && _last_activated[u] != _round) {
        _next_active_nodes.atomic_push_back(u);
      }
    }

    const NodeID v_height = u_height + 1;
    _graph->neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
      if (_node_status.is_terminal(v) || -_flow[e] == c || _heights[v] != max_level) {
        return;
      }

      if (__atomic_exchange_n(&_heights[v], v_height, __ATOMIC_ACQ_REL) == max_level) {
        queue.push_back(v);
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

void ParallelBlockedPreflowPushAlgorithm::discharge_active_nodes() {
  IF_STATS _stats.num_rounds += 1;
  IF_STATS _stats.num_discharges += _active_nodes.size();
  IF_STATS _stats.num_parallel_discharges += _active_nodes.size();

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, _active_nodes.size()), [&](const auto &range) {
    BufferedVector<NodeID>::Buffer next_active_nodes = _next_active_nodes.local_buffer();
    std::size_t &local_work_amount = _work_ets.local();

    for (NodeID i = range.begin(), end = range.end(); i < end; ++i) {
      const NodeID u = _active_nodes[i];
      KASSERT(!_node_status.is_terminal(u));

      atomic_discharge(u, next_active_nodes, local_work_amount);
    }
  });

  _next_active_nodes.flush();
}

void ParallelBlockedPreflowPushAlgorithm::apply_updates() {
  tbb::parallel_for<std::size_t>(0, _active_nodes.size(), [&](const std::size_t i) {
    const NodeID u = _active_nodes[i];

    _active_node_state[u] = kNotModifiedState;
    _heights[u] = _next_heights[u];

    _excess[u] += _excess_delta[u];
    _excess_delta[u] = 0;
  });

  tbb::parallel_for<std::size_t>(0, _next_active_nodes.size(), [&](const std::size_t i) {
    const NodeID u = _next_active_nodes[i];

    _active_node_state[u] = kNotModifiedState;

    _excess[u] += _excess_delta[u];
    _excess_delta[u] = 0;
  });

  tbb::parallel_for<std::size_t>(0, _acitve_sink_nodes.size(), [&](const std::size_t i) {
    const NodeID sink = _acitve_sink_nodes[i];
    KASSERT(_node_status.is_sink(sink));

    const EdgeWeight excess_delta = _excess_delta[sink];
    KASSERT(excess_delta >= 0);

    __atomic_fetch_add(&_flow_value, excess_delta, __ATOMIC_RELAXED);
    _excess_delta[sink] = 0;
  });

  _acitve_sink_nodes.clear();
}

void ParallelBlockedPreflowPushAlgorithm::atomic_discharge(
    const NodeID u, BufferedVector<NodeID>::Buffer next_active_nodes, std::size_t &local_work_amount
) {
  const EdgeID first_edge = _graph->first_edge(u);
  const NodeID degree = _graph->degree(u);
  const NodeID num_nodes = _graph->n();

  const EdgeWeight initial_excess = _excess[u];
  EdgeWeight excess = initial_excess;

  NodeID initial_edge_offset = _cur_edge_offsets[u];
  NodeID edge_offset = initial_edge_offset;

  bool skipped = false;
  NodeID u_height = _heights[u];
  while (excess > 0 && u_height < num_nodes) {
    if (edge_offset == degree) [[unlikely]] {
      if (skipped || !update_active_node(u, kRelabeledState)) {
        if (__atomic_exchange_n(&_last_activated[u], _round, __ATOMIC_ACQ_REL) != _round) {
          next_active_nodes.push_back(u);

          IF_STATS _stats.num_relabel_conflicts.local() += 1;
        }

        break;
      }

      local_work_amount += degree;

      u_height = relabel(u);
      edge_offset = 0;
      continue;
    }

    const EdgeID e = first_edge + edge_offset++;
    const NodeID v = _graph->edge_target(e);

    const NodeID v_height = _heights[v];
    if (u_height <= v_height) {
      continue;
    }

    const EdgeWeight e_flow = _flow[e];
    const EdgeWeight e_capacity = _graph->edge_weight(e);

    const EdgeWeight residual_capacity = e_capacity - e_flow;
    const EdgeWeight flow = std::min(excess, residual_capacity);
    if (flow <= 0) {
      continue;
    }

    if (!update_active_node(v, kPushedState)) {
      if (!skipped) {
        skipped = true;
        initial_edge_offset = edge_offset - 1;
      }

      IF_STATS _stats.num_push_conflicts.local() += 1;
      continue;
    }

    const EdgeID e_reverse = _reverse_edges[e];
    _flow[e] += flow;
    _flow[e_reverse] -= flow;

    excess -= flow;
    __atomic_fetch_add(&_excess_delta[v], flow, __ATOMIC_RELAXED);

    if (_node_status.is_sink(v) &&
        __atomic_exchange_n(&_last_activated[v], _round, __ATOMIC_ACQ_REL) != _round) {
      _acitve_sink_nodes.push_back(v);
      continue;
    }

    if (!_node_status.is_terminal(v) &&
        __atomic_exchange_n(&_last_activated[v], _round, __ATOMIC_ACQ_REL) != _round) {
      next_active_nodes.push_back(v);
    }
  }

  _next_heights[u] = u_height;
  _cur_edge_offsets[u] = skipped ? initial_edge_offset : edge_offset;

  const EdgeWeight excess_delta = excess - initial_excess;
  __atomic_fetch_add(&_excess_delta[u], excess_delta, __ATOMIC_RELAXED);
}

void ParallelBlockedPreflowPushAlgorithm::discharge(const NodeID u) {
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
      _active_nodes.push_back(v);
      _last_activated[v] = _round;
    }
  }

  _cur_edge_offsets[u] = cur_edge_offset;
  _heights[u] = u_height;
}

NodeID ParallelBlockedPreflowPushAlgorithm::relabel(const NodeID u) {
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

bool ParallelBlockedPreflowPushAlgorithm::update_active_node(
    const NodeID u, const std::uint8_t desired_state
) {
  std::uint8_t expected = kNotModifiedState;
  return _active_node_state[u] == desired_state || __atomic_compare_exchange_n(
                                                       &_active_node_state[u],
                                                       &expected,
                                                       desired_state,
                                                       false,
                                                       __ATOMIC_ACQ_REL,
                                                       __ATOMIC_RELAXED
                                                   );
}

std::span<const NodeID> ParallelBlockedPreflowPushAlgorithm::excess_nodes() {
  _excess_nodes.clear();

  for (const NodeID u : _graph->nodes()) {
    if (_excess[u] > 0 && !_node_status.is_terminal(u)) {
      _excess_nodes.push_back(u);
    }
  }

  return _excess_nodes;
}

const NodeStatus &ParallelBlockedPreflowPushAlgorithm::node_status() const {
  return _node_status;
}

void ParallelBlockedPreflowPushAlgorithm::free() {
  _node_status.free();

  _excess_nodes.clear();
  _excess_nodes.shrink_to_fit();

  _flow.free();

  _work_ets.clear();
  _parallel_bfs_runner.free();

  _nodes_to_desaturate.clear();
  _nodes_to_desaturate.shrink_to_fit();

  _last_activated.free();
  _cur_edge_offsets.free();
  _active_node_state.free();

  _excess.free();
  _excess_delta.free();

  _heights.free();
  _next_heights.free();

  _active_nodes.free();
  _next_active_nodes.free();
  _acitve_sink_nodes.clear();
}

} // namespace kaminpar::shm
