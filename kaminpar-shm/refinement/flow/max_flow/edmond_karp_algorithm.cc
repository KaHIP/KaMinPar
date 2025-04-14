#include "kaminpar-shm/refinement/flow/max_flow/edmond_karp_algorithm.h"

#include <algorithm>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

namespace {

struct PathEdge {
  NodeID from;
  EdgeID edge;
};

StaticArray<EdgeID> create_reverse_edges_index(const CSRGraph &graph) {
  StaticArray<EdgeID> reverse_edges(graph.m(), static_array::noinit);

  struct EdgeHasher {
    [[nodiscard]] std::size_t operator()(const std::pair<NodeID, NodeID> &edge) const noexcept {
      return edge.first ^ (edge.second << 1);
    }
  };
  std::unordered_map<std::pair<NodeID, NodeID>, EdgeID, EdgeHasher> edge_table;

  for (NodeID u = 0; u < graph.n(); u++) {
    graph.neighbors(u, [&](const EdgeID e, const NodeID v) {
      if (u < v) {
        edge_table.insert({{u, v}, e});
      } else {
        auto it = edge_table.find({v, u});
        KASSERT(it != edge_table.end());

        const EdgeID e_reverse = it->second;
        reverse_edges[e] = e_reverse;
        reverse_edges[e_reverse] = e;

        edge_table.erase(it);
      }
    });
  }

  return reverse_edges;
}

std::pair<NodeID, EdgeWeight> find_augmenting_path(
    const CSRGraph &graph,
    const std::unordered_set<NodeID> &sources,
    const std::unordered_set<NodeID> &sinks,
    std::span<const EdgeWeight> flow,
    std::span<PathEdge> predecessor
) {
  for (NodeID i = 0; i < graph.n(); i++) {
    predecessor[i] = {kInvalidNodeID, kInvalidEdgeID};
  }

  std::unordered_set<NodeID> visited;
  std::queue<std::pair<NodeID, EdgeWeight>> bfs_queue;
  for (const NodeID source : sources) {
    bfs_queue.emplace(source, std::numeric_limits<EdgeWeight>::max());
    predecessor[source] = {source, kInvalidEdgeID};
    visited.insert(source);
  }

  EdgeWeight net_flow = 0;
  NodeID sink = kInvalidNodeID;
  while (!bfs_queue.empty()) {
    const auto [u, u_flow] = bfs_queue.front();
    bfs_queue.pop();

    graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
      const bool was_visited = predecessor[v].from != kInvalidNodeID;
      if (was_visited) {
        return false;
      }

      const EdgeWeight residual_capacity = // Prevent overflow, TODO: different solution?
          (w == std::numeric_limits<EdgeWeight>::max() && flow[e] < 0)
              ? std::numeric_limits<EdgeWeight>::max()
              : w - flow[e];
      if (residual_capacity > 0) {
        predecessor[v] = {u, e};

        const EdgeWeight v_flow = std::min(u_flow, residual_capacity);
        if (sinks.contains(v)) {
          net_flow = v_flow;
          sink = v;
          return true;
        }

        bfs_queue.emplace(v, v_flow);
        visited.insert(v);
      }

      return false;
    });

    if (net_flow != 0) {
      break;
    }
  };

  return {sink, net_flow};
}

void augment_flow(
    const NodeID sink,
    const EdgeWeight net_flow,
    std::span<EdgeWeight> flow,
    std::span<PathEdge> predecessor,
    std::span<EdgeID> reverse_edges
) {
  NodeID cur = sink;

  while (predecessor[cur].from != cur) {
    const auto [prev, edge] = predecessor[cur];

    flow[edge] += net_flow;
    flow[reverse_edges[edge]] -= net_flow;

    cur = prev;
  }
}

} // namespace

void EdmondsKarpAlgorithm::compute(
    const CSRGraph &graph,
    const std::unordered_set<NodeID> &sources,
    const std::unordered_set<NodeID> &sinks,
    std::span<EdgeWeight> flow
) {
  KASSERT(
      debug::are_terminals_disjoint(sources, sinks),
      "source and sink nodes are not disjoint",
      assert::heavy
  );

  KASSERT(
      debug::is_valid_flow(graph, sources, sinks, flow),
      "given an invalid flow as basis",
      assert::heavy
  );

  StaticArray<PathEdge> predecessor(graph.n(), static_array::noinit);
  StaticArray<EdgeID> reverse_edges = create_reverse_edges_index(graph);

  while (true) {
    auto [sink, net_flow] = find_augmenting_path(graph, sources, sinks, flow, predecessor);

    if (net_flow == 0) {
      break;
    }

    augment_flow(sink, net_flow, flow, predecessor, reverse_edges);
  }

  IF_DBG debug::print_flow(graph, sources, sinks, flow);

  KASSERT(
      debug::is_valid_flow(graph, sources, sinks, flow),
      "computed an invalid flow using edmond-karp",
      assert::heavy
  );

  KASSERT(
      debug::is_max_flow(graph, sources, sinks, flow),
      "computed a non-maximum flow using edmond-karp",
      assert::heavy
  );
}

} // namespace kaminpar::shm
