#include "kaminpar-shm/refinement/flow/max_flow/preflow_push_algorithm.h"

#include <algorithm>
#include <queue>
#include <utility>

#include "kaminpar-shm/refinement/flow/util/reverse_edge_index.h"

namespace kaminpar::shm {

PreflowPushAlgorithm::PreflowPushAlgorithm(const PreflowPushContext &ctx) : _ctx(ctx) {}

void PreflowPushAlgorithm::initialize(const CSRGraph &graph) {
  _graph = &graph;
  _reverse_edge_index = compute_reverse_edge_index(graph);

  _grt = GlobalRelabelingThreshold(graph.m(), _ctx.global_relabeling_frequency);

  if (_flow.size() != graph.m()) {
    _flow.resize(graph.m());
  }

  if (_heights.size() < graph.n()) {
    _heights.resize(graph.n(), static_array::noinit);
  }

  if (_excess.size() < graph.n()) {
    _excess.resize(graph.n(), static_array::noinit);
  }

  if (_cur_edge_offsets.size() < graph.n()) {
    _cur_edge_offsets.resize(graph.n(), static_array::noinit);
  }

  if (_levels.size() < graph.n() * 2) {
    _levels.resize(graph.n() * 2);
  }
}

std::span<const EdgeWeight> PreflowPushAlgorithm::compute_max_flow(
    const std::unordered_set<NodeID> &sources, const std::unordered_set<NodeID> &sinks
) {
  KASSERT(
      debug::are_terminals_disjoint(sources, sinks),
      "source and sink nodes are not disjoint",
      assert::heavy
  );

  KASSERT(
      debug::is_valid_flow(*_graph, sources, sinks, _flow),
      "given an invalid flow as basis",
      assert::heavy
  );

  _sources = &sources;
  _sinks = &sinks;

  saturate_source_edges();
  compute_exact_heights();

  NodeID height = initialize_levels();
  while (height > 0) {
    while (true) {
      Level &level = _levels[height];
      if (!level.has_active_node()) {
        height -= 1;
        break;
      }

      const NodeID u = level.pop_activate_node();
      const NodeID u_height = discharge(u);

      if (_ctx.global_relabeling_heuristic && _grt.is_reached()) {
        height = global_relabel();
      } else {
        height = u_height;
      }
    }
  }

  IF_DBG debug::print_flow(*_graph, sources, sinks, _flow);

  KASSERT(
      debug::is_valid_flow(*_graph, sources, sinks, _flow),
      "computed an invalid flow using preflow-push",
      assert::heavy
  );

  KASSERT(
      debug::is_max_flow(*_graph, sources, sinks, _flow),
      "computed a non-maximum flow using preflow-push",
      assert::heavy
  );

  return _flow;
}

NodeID PreflowPushAlgorithm::global_relabel() {
  _grt.clear();

  compute_exact_heights();
  for (Level &level : _levels) {
    level.clear();
  }

  const NodeID max_active_height = initialize_levels();
  return max_active_height;
}

NodeID PreflowPushAlgorithm::initialize_levels() {
  const NodeID num_nodes = _graph->n();

  NodeID max_active_height = 0;
  for (const NodeID u : _graph->nodes()) {
    if (_sources->contains(u) || _sinks->contains(u)) {
      continue;
    }

    NodeID height = _heights[u];

    const bool is_unreachable_from_sink = height == kInvalidNodeID;
    if (is_unreachable_from_sink) {
      height = num_nodes + 1;
      _heights[u] = height;
    }

    const bool is_active = _excess[u] > 0;
    Level &level = _levels[height];
    if (is_active) {
      max_active_height = std::max(max_active_height, height);
      level.add_active_node(u);
    } else {
      level.add_inactive_node(u);
    }
  }

  return max_active_height;
}

void PreflowPushAlgorithm::saturate_source_edges() {
  std::fill(_cur_edge_offsets.begin(), _cur_edge_offsets.end(), 0);
  std::fill(_excess.begin(), _excess.end(), 0);

  for (const NodeID source : *_sources) {
    _excess[source] = std::numeric_limits<EdgeWeight>::max();
    _graph->neighbors(source, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
      const EdgeWeight residual_capacity = c - _flow[e];
      push(source, v, e, residual_capacity);
    });
  }
}

void PreflowPushAlgorithm::compute_exact_heights() {
  std::fill(_heights.begin(), _heights.end(), kInvalidNodeID);

  std::queue<std::pair<NodeID, NodeID>> bfs_queue;
  for (const NodeID sink : *_sinks) {
    _heights[sink] = 0;
    bfs_queue.emplace(sink, 0);
  }

  while (!bfs_queue.empty()) {
    const auto [u, u_height] = bfs_queue.front();
    bfs_queue.pop();

    const NodeID v_height = u_height + 1;
    _graph->neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
      if (_heights[v] != kInvalidNodeID || -_flow[e] == c) {
        return;
      }

      _heights[v] = v_height;
      bfs_queue.emplace(v, v_height);
    });
  }

  const NodeID num_nodes = _graph->n();
  for (const NodeID source : *_sources) {
    _heights[source] = num_nodes;
  }
}

NodeID PreflowPushAlgorithm::discharge(const NodeID u) {
  const EdgeID first_edge = _graph->first_edge(u);
  const NodeID degree = _graph->degree(u);

  NodeID u_height = _heights[u];
  while (_excess[u] > 0) {
    const EdgeID cur_edge_offset = _cur_edge_offsets[u];

    if (cur_edge_offset == degree) {
      u_height = relabel(u);
      _cur_edge_offsets[u] = 0;
    } else {
      const EdgeID e = first_edge + cur_edge_offset;
      const NodeID v = _graph->edge_target(e);
      const EdgeWeight e_capacity = _graph->edge_weight(e);

      const EdgeWeight e_flow = _flow[e];
      const EdgeWeight residual_capacity = // Prevent overflow, TODO: different solution?
          (e_capacity == std::numeric_limits<EdgeWeight>::max() && e_flow < 0)
              ? std::numeric_limits<EdgeWeight>::max()
              : e_capacity - e_flow;

      if (residual_capacity > 0) {
        const NodeID v_height = _heights[v];

        if (u_height > v_height) {
          const bool v_was_inactive = push(u, v, e, residual_capacity);

          if (v_was_inactive && !_sources->contains(v) && !_sinks->contains(v)) {
            _levels[v_height].activate(v);
          }
        }
      }

      _cur_edge_offsets[u] += 1;
    }
  }

  return u_height;
}

bool PreflowPushAlgorithm::push(
    const NodeID from, const NodeID to, const EdgeID e, const EdgeWeight residual_capacity
) {
  const EdgeWeight flow =
      _sources->contains(from) ? residual_capacity : std::min(_excess[from], residual_capacity);
  if (flow == 0) {
    return false;
  }

  _flow[e] += flow;
  _flow[_reverse_edge_index[e]] -= flow;

  const EdgeWeight to_prev_excess = _excess[to];
  _excess[to] = to_prev_excess + flow;
  _excess[from] -= flow;

  const bool to_was_inactive = to_prev_excess == 0;
  return to_was_inactive;
}

NodeID PreflowPushAlgorithm::relabel(const NodeID u) {
  _grt.add_work(_graph->degree(u));

  const NodeID old_height = _heights[u];
  _levels[old_height].remove(u);

  NodeID min_neighboring_height = std::numeric_limits<NodeID>::max();
  _graph->neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
    if (_flow[e] < c) {
      min_neighboring_height = std::min(min_neighboring_height, _heights[v]);
    }
  });

  KASSERT(min_neighboring_height != std::numeric_limits<NodeID>::max());
  const NodeID new_height = min_neighboring_height + 1;

  _heights[u] = new_height;
  _levels[new_height].add_inactive_node(u);

  return new_height;
}

} // namespace kaminpar::shm
