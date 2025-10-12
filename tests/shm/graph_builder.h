#pragma once

#include <memory>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "kassert/kassert.hpp"

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::testing {

class GraphBuilder {
public:
  GraphBuilder() = default;

  GraphBuilder(const NodeID n, const EdgeID m) {
    _nodes.reserve(n + 1);
    _edges.reserve(m);
    _node_weights.reserve(n);
    _edge_weights.reserve(m);
  }

  GraphBuilder(const GraphBuilder &) = delete;
  GraphBuilder &operator=(const GraphBuilder &) = delete;

  GraphBuilder(GraphBuilder &&) noexcept = default;
  GraphBuilder &operator=(GraphBuilder &&) noexcept = default;

  NodeID new_node(const NodeWeight weight = 1) {
    _nodes.push_back(_edges.size());
    _node_weights.push_back(weight);
    return _nodes.size() - 1;
  }

  NodeWeight &last_node_weight() {
    return _node_weights.back();
  }

  EdgeID new_edge(const NodeID v, const EdgeID weight = 1) {
    _edges.push_back(v);
    _edge_weights.push_back(weight);
    return _edges.size() - 1;
  }

  EdgeWeight &last_edge_weight() {
    return _edge_weights.back();
  }

  template <typename... Args> Graph build(Args &&...args) {
    _nodes.push_back(_edges.size());
    return Graph(std::make_unique<CSRGraph>(
        static_array::create(_nodes),
        static_array::create(_edges),
        static_array::create(_node_weights),
        static_array::create(_edge_weights),
        std::forward<Args>(args)...
    ));
  }

private:
  std::vector<EdgeID> _nodes;
  std::vector<NodeID> _edges;
  std::vector<NodeWeight> _node_weights;
  std::vector<EdgeWeight> _edge_weights;
};

class EdgeBasedGraphBuilder {
public:
  EdgeBasedGraphBuilder() : _num_edges(0) {}

  void add_edge(const NodeID u, const NodeID v, const EdgeWeight w = 1) {
    KASSERT(u != v, assert::always);

    _num_edges += 2;
    _neighborhoods[u].emplace_back(v, w);
    _neighborhoods[v].emplace_back(u, w);
  }

  [[nodiscard]] Graph build() const {
    const NodeID num_nodes = _neighborhoods.size();
    StaticArray<EdgeID> nodes(num_nodes + 1, static_array::noinit);

    StaticArray<NodeID> edges(_num_edges, static_array::noinit);
    StaticArray<EdgeWeight> edge_weights(_num_edges, static_array::noinit);

    EdgeID cur_edge = 0;
    bool has_unit_edge_weights = true;
    for (NodeID u = 0; u < num_nodes; ++u) {
      const auto &neighborhood = _neighborhoods.at(u);
      nodes[u + 1] = neighborhood.size();

      for (const auto &[v, w] : neighborhood) {
        edges[cur_edge] = v;
        edge_weights[cur_edge] = w;

        cur_edge += 1;
        has_unit_edge_weights &= w == 1;
      }
    }

    nodes[0] = 0;
    std::partial_sum(nodes.begin(), nodes.end(), nodes.begin());

    if (has_unit_edge_weights) {
      edge_weights.free();
    }

    return Graph(std::make_unique<CSRGraph>(
        std::move(nodes), std::move(edges), StaticArray<NodeWeight>(), std::move(edge_weights)
    ));
  }

private:
  EdgeID _num_edges;
  std::unordered_map<NodeID, std::vector<std::pair<NodeID, EdgeWeight>>> _neighborhoods;
};

} // namespace kaminpar::shm::testing
