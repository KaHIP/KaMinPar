#include "kaminpar-shm/refinement/flow/max_flow/highest_level_preflow_push_algorithm.h"

#include <algorithm>
#include <queue>
#include <utility>

#include "kaminpar-shm/refinement/flow/util/reverse_edge_index.h"

#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

HighestLevelPreflowPushAlgorithm::HighestLevelPreflowPushAlgorithm(
    const HighestLevelPreflowPushContext &ctx
)
    : _ctx(ctx) {}

void HighestLevelPreflowPushAlgorithm::initialize(const CSRGraph &graph) {
  _graph = &graph;
  _reverse_edge_index = compute_reverse_edge_index(graph);

  _flow_value = 0;

  if (_flow.size() != graph.m()) {
    _flow.resize(graph.m(), static_array::noinit);
  }
  std::fill(_flow.begin(), _flow.end(), 0);

  _grt = GlobalRelabelingThreshold(graph.m(), _ctx.global_relabeling_frequency);

  if (_excess.size() < graph.n()) {
    _excess.resize(graph.n(), static_array::noinit);
  }
  std::fill(_excess.begin(), _excess.end(), 0);

  if (_cur_edge_offsets.size() < graph.n()) {
    _cur_edge_offsets.resize(graph.n(), static_array::noinit);
  }

  if (_heights.size() < graph.n()) {
    _heights.resize(graph.n(), static_array::noinit);
  }

  _levels.resize(graph.n() * 2);
}

MaxFlowAlgorithm::Result HighestLevelPreflowPushAlgorithm::compute_max_flow(
    const std::unordered_set<NodeID> &sources, const std::unordered_set<NodeID> &sinks
) {
  KASSERT(
      debug::are_terminals_disjoint(*_sources, *_sinks),
      "source and sink nodes are not disjoint",
      assert::heavy
  );

  _sources = &sources;
  _sinks = &sinks;

  // Initialize the preflow by saturating all edges emanating from the sources.
  saturate_source_edges();

  if (_ctx.two_phase) {
    // Phase 1: Find a maximum preflow by pushing as much flow as possible to the terminals.
    find_maximum_preflow();

    // Phase 2: Convert maximum preflow into maximum flow by returning the excess to the sources.
    convert_maximum_preflow();
  } else {
    find_maximum_flow();
  }

  IF_DBG debug::print_flow(*_graph, *_sources, *_sinks, _flow);

  KASSERT(
      debug::is_valid_flow(*_graph, *_sources, *_sinks, _flow),
      "computed an invalid flow using preflow-push",
      assert::heavy
  );

  KASSERT(
      debug::is_max_flow(*_graph, *_sources, *_sinks, _flow),
      "computed a non-maximum flow using preflow-push",
      assert::heavy
  );

  return Result(_flow_value, _flow);
}

void HighestLevelPreflowPushAlgorithm::saturate_source_edges() {
  SCOPED_TIMER("Saturating Source Edges");

  std::fill(_cur_edge_offsets.begin(), _cur_edge_offsets.end(), 0);

  for (const NodeID source : *_sources) {
    _graph->neighbors(source, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
      const EdgeWeight residual_capacity = c - _flow[e];
      push(source, v, e, residual_capacity);
    });
  }
}

void HighestLevelPreflowPushAlgorithm::find_maximum_preflow() {
  NodeID height = global_relabel<true>();
  while (height > 0) {
    while (true) {
      Level &level = _levels[height];
      if (!level.has_active_node()) {
        height -= 1;
        break;
      }

      const NodeID u = level.pop_activate_node();
      const NodeID u_height = discharge<true>(u, height);

      if (false && _ctx.gap_heuristic && level.emtpy()) {
        employ_gap_heuristic(height);

        height = height - 1;
      } else if (_ctx.global_relabeling_heuristic && _grt.is_reached()) {
        height = global_relabel<true>();
      } else {
        height = u_height;
      }
    }
  }
}

void HighestLevelPreflowPushAlgorithm::convert_maximum_preflow() {
  const NodeID n = _graph->n();

  NodeID height = global_relabel<false>();
  while (height > n) {
    while (true) {
      Level &level = _levels[height];
      if (!level.has_active_node()) {
        height -= 1;
        break;
      }

      const NodeID u = level.pop_activate_node();
      const NodeID u_height = discharge<false>(u, height);

      if (_ctx.global_relabeling_heuristic && _grt.is_reached()) {
        height = global_relabel<false>();
        break;
      }

      height = u_height;
    }
  }
}

void HighestLevelPreflowPushAlgorithm::find_maximum_flow() {
  NodeID height = global_relabel<true>();
  while (height > 0) {
    while (true) {
      Level &level = _levels[height];
      if (!level.has_active_node()) {
        height -= 1;
        break;
      }

      const NodeID u = level.pop_activate_node();
      height = discharge<false>(u, height);

      if (_ctx.global_relabeling_heuristic && _grt.is_reached()) {
        height = global_relabel<false>();
        break;
      }
    }
  }
}

template <bool kFirstPhase> NodeID HighestLevelPreflowPushAlgorithm::global_relabel() {
  SCOPED_TIMER("Global Relabeling");
  _grt.clear();

  const std::unordered_set<NodeID> &terminals = kFirstPhase ? *_sinks : *_sources;
  compute_exact_heights(terminals);

  const NodeID max_active_height = initialize_levels<kFirstPhase>();
  return max_active_height;
}

void HighestLevelPreflowPushAlgorithm::compute_exact_heights(
    const std::unordered_set<NodeID> &terminals
) {
  const NodeID max_level = 2 * _graph->n();
  std::fill(_heights.begin(), _heights.end(), max_level);

  std::queue<std::pair<NodeID, NodeID>> bfs_queue;
  for (const NodeID terminal : terminals) {
    _heights[terminal] = 0;
    bfs_queue.emplace(terminal, 0);
  }

  while (!bfs_queue.empty()) {
    const auto [u, u_height] = bfs_queue.front();
    bfs_queue.pop();

    const NodeID v_height = u_height + 1;
    _graph->neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
      if (_sources->contains(v) || _sinks->contains(v) || _heights[v] != max_level ||
          -_flow[e] == c) {
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

template <bool kFirstPhase> NodeID HighestLevelPreflowPushAlgorithm::initialize_levels() {
  const NodeID num_nodes = _graph->n();

  for (Level &level : _levels) {
    level.clear();
  }

  NodeID max_active_height = 0;
  for (const NodeID u : _graph->nodes()) {
    if (_sources->contains(u) || _sinks->contains(u)) {
      continue;
    }

    NodeID height = _heights[u];

    if constexpr (kFirstPhase) {
      const bool is_reachable_from_sink = height != kInvalidNodeID;

      if (is_reachable_from_sink) {
        max_active_height = std::max(max_active_height, height);
      } else {
        height = num_nodes + 1;
        _heights[u] = height;
      }
    } else {
      // Shift the computed heights because the height of the sources is n.
      height += num_nodes;
      _heights[u] = height;

      max_active_height = std::max(max_active_height, height);

      // TODO: look if height == kInvalidNodeID might be an issue here.
      KASSERT(height != kInvalidNodeID);
    }

    const bool is_active = _excess[u] > 0;
    Level &level = _levels[height];
    if (is_active) {
      level.add_active_node(u);
    } else {
      level.add_inactive_node(u);
    }
  }

  return max_active_height;
}

void HighestLevelPreflowPushAlgorithm::employ_gap_heuristic(NodeID height) {
  SCOPED_TIMER("Gap Heuristic");
  const NodeID np1 = _graph->n() + 1;

  Level level_np1 = _levels[np1];
  while (++height < np1) {
    Level &level = _levels[height];

    for (const NodeID u : level.active_nodes()) {
      _heights[u] = np1;
      level_np1.add_active_node(u);
    }

    for (const NodeID u : level.inactive_nodes()) {
      _heights[u] = np1;
      level_np1.add_inactive_node(u);
    }

    level.clear();
  }
}

template <bool kFirstPhase>
NodeID HighestLevelPreflowPushAlgorithm::discharge(const NodeID u, NodeID u_height) {
  SCOPED_TIMER("Discharging");

  const EdgeID first_edge = _graph->first_edge(u);
  const NodeID degree = _graph->degree(u);

  while (_excess[u] > 0) {
    const EdgeID cur_edge_offset = _cur_edge_offsets[u];

    if (cur_edge_offset == degree) {
      _grt.add_work(degree);

      _cur_edge_offsets[u] = 0;
      u_height = relabel(u, u_height);

      // Although the node is still active, with a height at least n - 1,
      // it is now known to be on the source-side of the minimum cut. Thus,
      // stop processing it until phase 2.
      if (kFirstPhase && u_height >= _graph->n() - 1) {
        _levels[u_height].activate(u);
        break;
      }
    } else {
      const EdgeID e = first_edge + cur_edge_offset;
      const NodeID v = _graph->edge_target(e);

      const EdgeWeight e_flow = _flow[e];
      const EdgeWeight e_capacity = _graph->edge_weight(e);

      const EdgeWeight residual_capacity = e_capacity - e_flow;
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

bool HighestLevelPreflowPushAlgorithm::push(
    const NodeID from, const NodeID to, const EdgeID e, const EdgeWeight residual_capacity
) {
  const bool from_source = _sources->contains(from);
  const EdgeWeight flow =
      from_source ? residual_capacity : std::min(_excess[from], residual_capacity);
  if (flow == 0) {
    return false;
  }

  _flow[e] += flow;
  _flow[_reverse_edge_index[e]] -= flow;

  const EdgeWeight to_prev_excess = _excess[to];
  _excess[to] = to_prev_excess + flow;
  _excess[from] -= flow;

  if (from_source) {
    _flow_value += flow;
  } else if (_sources->contains(to)) {
    _flow_value -= flow;
  }

  const bool to_was_inactive = to_prev_excess == 0;
  return to_was_inactive;
}

NodeID HighestLevelPreflowPushAlgorithm::relabel(const NodeID u, const NodeID u_height) {
  _levels[u_height].remove_inactive_node(u);

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
  _levels[new_height].add_inactive_node(u);

  return new_height;
}

} // namespace kaminpar::shm
