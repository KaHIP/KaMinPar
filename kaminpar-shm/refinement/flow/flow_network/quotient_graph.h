#pragma once

#include <span>

#include <tbb/concurrent_vector.h>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm {

class QuotientGraph {
public:
  struct GraphEdge {
    NodeID u;
    NodeID v;
  };

  struct Edge {
    Edge() = default;

    Edge(const Edge &) = delete;
    Edge &operator=(const Edge &) = delete;

    Edge(Edge &&) noexcept = default;
    Edge &operator=(Edge &&) noexcept = default;

    mutable tbb::concurrent_vector<GraphEdge> cut_edges;
    EdgeWeight cut_weight;
    EdgeWeight total_gain;
  };

public:
  QuotientGraph(const PartitionedCSRGraph &p_graph)
      : _p_graph(p_graph),
        _edges(p_graph.k() * p_graph.k()) {
    for (Edge &edge : _edges) {
      edge.cut_weight = 0;
      edge.total_gain = 0;
    }

    reconstruct();
  }

  QuotientGraph(const QuotientGraph &) = delete;
  QuotientGraph &operator=(const QuotientGraph &) = delete;

  QuotientGraph(QuotientGraph &&) noexcept = default;
  QuotientGraph &operator=(QuotientGraph &&) noexcept = default;

  void reconstruct();

  void add_gain(BlockID b1, BlockID b2, EdgeWeight gain);

  void add_cut_edges(std::span<const GraphEdge> new_cut_edges);

  [[nodiscard]] EdgeWeight total_cut_weight() const {
    return _total_cut_weight;
  }

  [[nodiscard]] bool has_quotient_edge(const BlockID b1, const BlockID b2) const {
    const Edge &quotient_edge = edge(b1, b2);
    return quotient_edge.cut_weight > 0;
  }

  [[nodiscard]] EdgeWeight quotient_edge_weight(const BlockID b1, const BlockID b2) const {
    const Edge &quotient_edge = edge(b1, b2);
    return quotient_edge.cut_weight;
  }

  template <typename Lambda>
  void foreach_cut_edge_shuffled(const BlockID b1, const BlockID b2, Lambda &&l) const {
    KASSERT(b1 < b2);

    const Edge &quotient_edge = edge(b1, b2);
    const std::size_t num_cut_edges = quotient_edge.cut_edges.size();

    Random::instance().shuffle(
        quotient_edge.cut_edges.begin(), quotient_edge.cut_edges.begin() + num_cut_edges
    );

    for (std::size_t i = 0; i < num_cut_edges; ++i) {
      QuotientGraph::GraphEdge edge = quotient_edge.cut_edges[i];
      l(edge.u, edge.v);
    }
  }

  [[nodiscard]] const Edge &edge(const BlockID b1, const BlockID b2) const {
    KASSERT(b1 != b2);
    KASSERT(b1 < _p_graph.k());
    KASSERT(b2 < _p_graph.k());

    return _edges[std::min(b1, b2) * _p_graph.k() + std::max(b1, b2)];
  }

  [[nodiscard]] Edge &edge(const BlockID b1, const BlockID b2) {
    KASSERT(b1 != b2);
    KASSERT(b1 < _p_graph.k());
    KASSERT(b2 < _p_graph.k());

    return _edges[std::min(b1, b2) * _p_graph.k() + std::max(b1, b2)];
  }

private:
  const PartitionedCSRGraph &_p_graph;

  EdgeWeight _total_cut_weight;
  ScalableVector<Edge> _edges;
};

} // namespace kaminpar::shm
