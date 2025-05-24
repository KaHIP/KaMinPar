#include "kaminpar-shm/refinement/flow/max_flow/edmond_karp_algorithm.h"

#include <algorithm>
#include <limits>
#include <queue>

#include "kaminpar-common/assert.h"

namespace kaminpar::shm {

void EdmondsKarpAlgorithm::initialize(
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

  if (_predecessor.size() < graph.n()) {
    _predecessor.resize(graph.n(), static_array::noinit);
  }
}

void EdmondsKarpAlgorithm::add_sources([[maybe_unused]] std::span<const NodeID> sources) {
  for (const NodeID u : sources) {
    KASSERT(!_node_status.is_sink(u));

    if (_node_status.is_unknown(u)) {
      _node_status.add_source(u);
    }
  }
}

void EdmondsKarpAlgorithm::add_sinks([[maybe_unused]] std::span<const NodeID> sinks) {
  for (const NodeID u : sinks) {
    KASSERT(!_node_status.is_source(u));

    if (_node_status.is_unknown(u)) {
      _node_status.add_sink(u);
    }
  }
}

void EdmondsKarpAlgorithm::pierce_nodes(std::span<const NodeID> nodes, const bool source_side) {
  if (source_side) {
    add_sources(nodes);
  } else {
    add_sinks(nodes);
  }
}

MaxFlowAlgorithm::Result EdmondsKarpAlgorithm::compute_max_flow() {
  while (true) {
    auto [sink, net_flow] = find_augmenting_path();

    if (net_flow == 0) {
      break;
    }

    augment_flow(sink, net_flow);
  }

  IF_DBG debug::print_flow(*_graph, _node_status, _flow);

  KASSERT(
      debug::is_valid_flow(*_graph, _node_status, _flow),
      "computed an invalid flow using edmond-karp",
      assert::heavy
  );

  KASSERT(
      debug::is_max_flow(*_graph, _node_status, _flow),
      "computed a non-maximum flow using edmond-karp",
      assert::heavy
  );

  KASSERT(
      _flow_value == debug::flow_value(*_graph, _node_status, _flow),
      "computed an invalid flow value using edmond-karp",
      assert::heavy
  );

  return Result(_flow_value, _flow);
}

const NodeStatus &EdmondsKarpAlgorithm::node_status() const {
  return _node_status;
}

std::pair<NodeID, EdgeWeight> EdmondsKarpAlgorithm::find_augmenting_path() {
  for (NodeID i = 0; i < _graph->n(); i++) {
    _predecessor[i] = {kInvalidNodeID, kInvalidEdgeID};
  }

  std::queue<std::pair<NodeID, EdgeWeight>> bfs_queue;
  for (const NodeID source : _node_status.source_nodes()) {
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

      const EdgeWeight residual_capacity = w - _flow[e];
      if (residual_capacity > 0) {
        _predecessor[v] = {u, e};

        const EdgeWeight v_flow = std::min(u_flow, residual_capacity);
        if (_node_status.is_sink(v)) {
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
    _flow[_reverse_edges[edge]] -= net_flow;

    cur = prev;
  }

  _flow_value += net_flow;
}

} // namespace kaminpar::shm
