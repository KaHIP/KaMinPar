#include "kaminpar-shm/refinement/flow/multiway_cut/isolating_cut_heuristic.h"

#include <limits>
#include <queue>
#include <unordered_set>
#include <utility>

#include "kaminpar-shm/refinement/flow/max_flow/edmond_karp_algorithm.h"
#include "kaminpar-shm/refinement/flow/max_flow/fifo_preflow_push_algorithm.h"
#include "kaminpar-shm/refinement/flow/max_flow/highest_level_preflow_push_algorithm.h"
#include "kaminpar-shm/refinement/flow/util/reverse_edge_index.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

IsolatingCutHeuristic::IsolatingCutHeuristic(const IsolatingCutHeuristicContext &ctx) : _ctx(ctx) {
  switch (ctx.flow_algorithm) {
  case FlowAlgorithm::EDMONDS_KARP:
    _max_flow_algorithm = std::make_unique<EdmondsKarpAlgorithm>();
    break;
  case FlowAlgorithm::FIFO_PREFLOW_PUSH:
    _max_flow_algorithm = std::make_unique<FIFOPreflowPushAlgorithm>(ctx.fifo_preflow_push);
    break;
  case FlowAlgorithm::HIGHEST_LEVEL_PREFLOW_PUSH:
    _max_flow_algorithm =
        std::make_unique<HighestLevelPreflowPushAlgorithm>(ctx.highest_level_preflow_push);
    break;
  }
}

std::unordered_set<EdgeID> IsolatingCutHeuristic::compute(
    const CSRGraph &graph, const std::vector<std::unordered_set<NodeID>> &terminal_sets
) {
  _graph = &graph;
  _reverse_edge_index = compute_reverse_edge_index(graph);

  std::unordered_set<EdgeID> cut_edges;

  std::unordered_set<NodeID> other_terminals;
  for (const std::unordered_set<NodeID> &terminals : terminal_sets) {
    for (const NodeID terminal : terminals) {
      other_terminals.insert(terminal);
    }
  }

  Cut cur_max_weighted_cut(std::numeric_limits<EdgeWeight>::min(), {});

  _max_flow_algorithm->initialize(graph);
  for (const std::unordered_set<NodeID> &terminals : terminal_sets) {
    for (const NodeID terminal : terminals) {
      other_terminals.erase(terminal);
    }

    auto [flow_value, flow] = TIMED_SCOPE("Compute Max Flow") {
      return _max_flow_algorithm->compute_max_flow(terminals, other_terminals);
    };

    Cut cut = compute_cut(terminals, flow);
    KASSERT(flow_value == cut.value);

    if (cut.value > cur_max_weighted_cut.value) {
      std::swap(cut, cur_max_weighted_cut);
    }

    for (const EdgeID cut_edge : cut.edges) {
      cut_edges.insert(cut_edge);
    }

    for (const NodeID terminal : terminals) {
      other_terminals.insert(terminal);
    }
  }

  KASSERT(
      debug::is_valid_multiway_cut(graph, terminal_sets, cut_edges),
      "computed a non-valid multi-way cut using the isolating-cut heuristic",
      assert::heavy
  );

  return cut_edges;
}

IsolatingCutHeuristic::Cut IsolatingCutHeuristic::compute_cut(
    const std::unordered_set<NodeID> &terminals, std::span<const EdgeWeight> flow
) {
  std::queue<NodeID> bfs_queue;
  for (const NodeID terminal : terminals) {
    bfs_queue.push(terminal);
  }

  std::unordered_set<NodeID> terminal_side_nodes;
  while (!bfs_queue.empty()) {
    const NodeID u = bfs_queue.front();
    bfs_queue.pop();

    _graph->neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
      if (terminal_side_nodes.contains(v) || flow[e] == w) {
        return;
      }

      terminal_side_nodes.insert(v);
      bfs_queue.push(v);
    });
  }

  EdgeWeight cut_value = 0;
  std::unordered_set<EdgeID> cut_edges;
  for (const NodeID u : terminal_side_nodes) {
    _graph->neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
      if (terminal_side_nodes.contains(v)) {
        return;
      }

      cut_value += w;
      cut_edges.insert(e);
      cut_edges.insert(_reverse_edge_index[e]);
    });
  }

  return Cut(cut_value, std::move(cut_edges));
}

} // namespace kaminpar::shm
