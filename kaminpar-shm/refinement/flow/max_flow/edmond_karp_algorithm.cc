#include "kaminpar-shm/refinement/flow/max_flow/edmond_karp_algorithm.h"

#include <algorithm>
#include <limits>
#include <queue>

#include "kaminpar-shm/refinement/flow/util/reverse_edge_index.h"

#include "kaminpar-common/assert.h"

namespace kaminpar::shm {

void EdmondsKarpAlgorithm::initialize(const CSRGraph &graph) {
  _graph = &graph;
  _reverse_edge_index = compute_reverse_edge_index(graph);

  if (_flow.size() != graph.m()) {
    _flow.resize(graph.m());
  }

  if (_predecessor.size() < graph.n()) {
    _predecessor.resize(graph.n(), static_array::noinit);
  }
}

std::span<const EdgeWeight> EdmondsKarpAlgorithm::compute_max_flow(
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

  while (true) {
    auto [sink, net_flow] = find_augmenting_path(sources, sinks);

    if (net_flow == 0) {
      break;
    }

    augment_flow(sink, net_flow);
  }

  IF_DBG debug::print_flow(*_graph, sources, sinks, _flow);

  KASSERT(
      debug::is_valid_flow(*_graph, sources, sinks, _flow),
      "computed an invalid flow using edmond-karp",
      assert::heavy
  );

  KASSERT(
      debug::is_max_flow(*_graph, sources, sinks, _flow),
      "computed a non-maximum flow using edmond-karp",
      assert::heavy
  );

  return _flow;
}

std::pair<NodeID, EdgeWeight> EdmondsKarpAlgorithm::find_augmenting_path(
    const std::unordered_set<NodeID> &sources, const std::unordered_set<NodeID> &sinks
) {
  for (NodeID i = 0; i < _graph->n(); i++) {
    _predecessor[i] = {kInvalidNodeID, kInvalidEdgeID};
  }

  std::queue<std::pair<NodeID, EdgeWeight>> bfs_queue;
  for (const NodeID source : sources) {
    bfs_queue.emplace(source, std::numeric_limits<EdgeWeight>::max());
    _predecessor[source] = {source, kInvalidEdgeID};
  }

  NodeID sink = kInvalidNodeID;
  EdgeWeight net_flow = 0;
  while (!bfs_queue.empty()) {
    const auto [u, u_flow] = bfs_queue.front();
    bfs_queue.pop();

    _graph->neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
      const bool was_visited = _predecessor[v].from != kInvalidNodeID;
      if (was_visited) {
        return false;
      }

      const EdgeWeight residual_capacity = // Prevent overflow, TODO: different solution?
          (w == std::numeric_limits<EdgeWeight>::max() && _flow[e] < 0)
              ? std::numeric_limits<EdgeWeight>::max()
              : w - _flow[e];
      if (residual_capacity > 0) {
        _predecessor[v] = {u, e};

        const EdgeWeight v_flow = std::min(u_flow, residual_capacity);
        if (sinks.contains(v)) {
          net_flow = v_flow;
          sink = v;
          return true;
        }

        bfs_queue.emplace(v, v_flow);
      }

      return false;
    });

    if (net_flow != 0) {
      break;
    }
  };

  return {sink, net_flow};
}

void EdmondsKarpAlgorithm::augment_flow(const NodeID sink, const EdgeWeight net_flow) {
  NodeID cur = sink;

  while (_predecessor[cur].from != cur) {
    const auto [prev, edge] = _predecessor[cur];

    _flow[edge] += net_flow;
    _flow[_reverse_edge_index[edge]] -= net_flow;

    cur = prev;
  }
}

} // namespace kaminpar::shm
