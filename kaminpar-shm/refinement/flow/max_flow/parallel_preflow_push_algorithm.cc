#include "kaminpar-shm/refinement/flow/max_flow/parallel_preflow_push_algorithm.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>

#include <oneapi/tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/refinement/flow/util/buffered_vector.h"

namespace kaminpar::shm {

ParallelPreflowPushAlgorithm::ParallelPreflowPushAlgorithm(const PreflowPushContext &ctx)
    : _ctx(ctx) {}

void ParallelPreflowPushAlgorithm::initialize(
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

  _round = 1;
  _grt = GlobalRelabelingThreshold(graph.n(), graph.m(), _ctx.global_relabeling_frequency);

  if (_last_touched.size() < graph.n()) {
    _last_touched.resize(graph.n(), static_array::noinit);
  }
  std::fill_n(_last_touched.begin(), graph.n(), 0);

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

  saturate_source_edges(_node_status.source_nodes());
  global_relabel();
  _next_active_nodes.flush();
}

void ParallelPreflowPushAlgorithm::add_sources(std::span<const NodeID> sources) {
  const NodeID num_nodes = _graph->n();

  for (const NodeID u : sources) {
    KASSERT(!_node_status.is_sink(u));

    if (_node_status.is_unknown(u)) {
      _node_status.add_source(u);
      _heights[u] = num_nodes;
    }
  }
}

void ParallelPreflowPushAlgorithm::add_sinks(std::span<const NodeID> sinks) {
  for (const NodeID u : sinks) {
    KASSERT(!_node_status.is_source(u));

    if (_node_status.is_unknown(u)) {
      _node_status.add_sink(u);
      _flow_value += _excess[u];
    }
  }
}

void ParallelPreflowPushAlgorithm::pierce_nodes(
    const bool source_side, std::span<const NodeID> nodes
) {
  KASSERT(_next_active_nodes.empty());
  _round += 1;

  if (source_side) {
    add_sources(nodes);
    saturate_source_edges(nodes);
  } else {
    add_sinks(nodes);
    saturate_source_edges(_node_status.source_nodes());

    constexpr bool kCollectActiveNodes = true;
    global_relabel<kCollectActiveNodes>();
  }

  _next_active_nodes.flush();

  KASSERT(
      debug::is_valid_labeling(*_graph, _node_status, _flow, _heights),
      "computed an invalid labeling using preflow-push",
      assert::heavy
  );
}

MaxPreflowAlgorithm::Result ParallelPreflowPushAlgorithm::compute_max_preflow() {
  while (!_next_active_nodes.empty()) {
    std::swap(_active_nodes, _next_active_nodes);
    _next_active_nodes.clear();
    _round += 1;

    discharge_active_nodes();
    apply_updates();

    _grt.add_work(_work_ets.combine(std::plus<>()));
    _work_ets.clear();

    if (_ctx.global_relabeling_heuristic && _grt.is_reached()) {
      global_relabel();
    }
  }

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

std::span<const NodeID> ParallelPreflowPushAlgorithm::excess_nodes() {
  _excess_nodes.clear();

  for (const NodeID u : _graph->nodes()) {
    if (_excess[u] > 0 && !_node_status.is_terminal(u)) {
      _excess_nodes.push_back(u);
    }
  }

  return _excess_nodes;
}

const NodeStatus &ParallelPreflowPushAlgorithm::node_status() const {
  return _node_status;
}

void ParallelPreflowPushAlgorithm::saturate_source_edges(std::span<const NodeID> sources) {
  tbb::parallel_for<std::size_t>(0, sources.size(), [&](const std::size_t i) {
    const NodeID source = sources[i];
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

      __atomic_fetch_add(&_excess[v], residual_capacity, __ATOMIC_RELAXED);

      if (_node_status.is_sink(v)) {
        __atomic_fetch_add(&_flow_value, residual_capacity, __ATOMIC_RELAXED);
      } else if (__atomic_exchange_n(&_last_touched[v], _round, __ATOMIC_ACQ_REL) != _round) {
        _next_active_nodes.atomic_push_back(v);
      }
    });
  });
}

template <bool kCollectActiveNodes> void ParallelPreflowPushAlgorithm::global_relabel() {
  _grt.clear();

  const NodeID num_nodes = _graph->n();
  const NodeID max_level = 2 * num_nodes;
  std::fill_n(_heights.begin(), num_nodes, max_level);

  _parallel_bfs_runner.reset(num_nodes);
  for (const NodeID sink : _node_status.sink_nodes()) {
    _heights[sink] = 0;
    _parallel_bfs_runner.add_seed(sink);
  }

  _parallel_bfs_runner.perform([&](const NodeID u, const NodeID u_height, auto &queue) {
    const NodeID v_height = u_height + 1;

    _graph->neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
      if (_node_status.is_terminal(v) || -_flow[e] == c || _heights[v] != max_level) {
        return;
      }

      if (__atomic_exchange_n(&_heights[v], v_height, __ATOMIC_ACQ_REL) != max_level) {
        return;
      }

      queue.push_back(v);

      if constexpr (kCollectActiveNodes) {
        if (_excess[v] > 0 && v_height < num_nodes &&
            __atomic_exchange_n(&_last_touched[v], _round, __ATOMIC_ACQ_REL) != _round) {
          _next_active_nodes.atomic_push_back(v);
        }
      }
    });
  });

  for (const NodeID source : _node_status.source_nodes()) {
    _heights[source] = num_nodes;
  }

  for (const NodeID u : _graph->nodes()) {
    if (_heights[u] == max_level) {
      _heights[u] = num_nodes + 1;
    }
  }
}

void ParallelPreflowPushAlgorithm::discharge_active_nodes() {
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, _active_nodes.size()), [&](const auto &range) {
    BufferedVector<NodeID>::Buffer next_active_nodes = _next_active_nodes.local_buffer();

    for (NodeID i = range.begin(), end = range.end(); i < end; ++i) {
      const NodeID u = _active_nodes[i];
      discharge(u, next_active_nodes);
    }
  });

  _next_active_nodes.flush();
}

void ParallelPreflowPushAlgorithm::apply_updates() {
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

  const std::span<const NodeID> sinks = _node_status.sink_nodes();
  tbb::parallel_for<std::size_t>(0, sinks.size(), [&](const std::size_t i) {
    const NodeID sink = sinks[i];

    const EdgeWeight excess_delta = _excess_delta[sink];
    KASSERT(excess_delta >= 0);

    __atomic_fetch_add(&_flow_value, excess_delta, __ATOMIC_RELAXED);
    _excess_delta[sink] = 0;
  });
}

void ParallelPreflowPushAlgorithm::discharge(
    const NodeID u, BufferedVector<NodeID>::Buffer next_active_nodes
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
    if (edge_offset == degree) {
      if (skipped || !update_active_node(u, kRelabeledState)) {
        if (__atomic_exchange_n(&_last_touched[u], _round, __ATOMIC_ACQ_REL) != _round) {
          next_active_nodes.push_back(u);
        }

        break;
      }

      _work_ets.local() += degree;

      u_height = relabel(u);
      edge_offset = 0;
      continue;
    }

    const EdgeID e = first_edge + edge_offset++;
    const NodeID v = _graph->edge_target(e);
    if (u_height <= _heights[v]) {
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

      continue;
    }

    const EdgeID e_reverse = _reverse_edges[e];
    _flow[e] += flow;
    _flow[e_reverse] -= flow;

    excess -= flow;
    __atomic_fetch_add(&_excess_delta[v], flow, __ATOMIC_RELAXED);

    if (!_node_status.is_terminal(v) &&
        __atomic_exchange_n(&_last_touched[v], _round, __ATOMIC_ACQ_REL) != _round) {
      next_active_nodes.push_back(v);
    }
  }

  _next_heights[u] = u_height;
  _cur_edge_offsets[u] = skipped ? initial_edge_offset : edge_offset;

  const EdgeWeight excess_delta = excess - initial_excess;
  __atomic_fetch_add(&_excess_delta[u], excess_delta, __ATOMIC_RELAXED);
}

NodeID ParallelPreflowPushAlgorithm::relabel(const NodeID u) {
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

bool ParallelPreflowPushAlgorithm::update_active_node(
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

} // namespace kaminpar::shm
