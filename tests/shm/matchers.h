#pragma once

#include <gmock/gmock.h>

#include "kaminpar-shm/datastructures/graph.h"

namespace kaminpar::shm::testing {
using ::testing::Matcher;
using ::testing::MatcherInterface;
using ::testing::MatchResultListener;

class HasEdgeMatcher : public MatcherInterface<const Graph &> {
public:
  HasEdgeMatcher(const NodeID u, const NodeID v) : _u(u), _v(v) {}

  bool MatchAndExplain(const Graph &graph, MatchResultListener *) const override {
    bool found_u_v = false;
    bool found_v_u = false;

    for (const NodeID v_prime : graph.adjacent_nodes(_u)) {
      if (_v == v_prime) {
        found_u_v = true;
        break;
      }
    }

    for (const NodeID u_prime : graph.adjacent_nodes(_v)) {
      if (_u == u_prime) {
        found_v_u = true;
        break;
      }
    }

    return found_u_v && found_v_u;
  }

  void DescribeTo(std::ostream *os) const override {
    *os << "graph has edge {" << _u << ", " << _v << "}";
  }

  void DescribeNegationTo(std::ostream *os) const override {
    *os << "graph does not have edge {" << _u << ", " << _v << "}";
  }

private:
  NodeID _u;
  NodeID _v;
};

inline Matcher<const Graph &> HasEdge(const NodeID u, const NodeID v) {
  return MakeMatcher(new HasEdgeMatcher(u, v));
}

class HasWeightedEdgeWithWeightedEndpointsMatcher : public MatcherInterface<const Graph &> {
public:
  HasWeightedEdgeWithWeightedEndpointsMatcher(
      const NodeWeight u_weight, const EdgeWeight e_weight, const NodeWeight v_weight
  )
      : _u_weight(u_weight),
        _e_weight(e_weight),
        _v_weight(v_weight) {}

  bool MatchAndExplain(const Graph &graph, MatchResultListener *) const override {
    for (const NodeID u : graph.nodes()) {
      if (graph.node_weight(u) == _u_weight) {
        for (const auto [e, v] : graph.neighbors(u)) {
          if ((_e_weight == 0 || graph.edge_weight(e) == _e_weight) &&
              graph.node_weight(v) == _v_weight) {
            return true;
          }
        }
      }
    }

    return false;
  }

  void DescribeTo(std::ostream *os) const override {
    *os << "graph has an edge with endpoints weighted " << _u_weight << " and " << _v_weight;
  }

  void DescribeNegationTo(std::ostream *os) const override {
    *os << "graph has no edge with endpoints weighted " << _u_weight << " and " << _v_weight;
  }

private:
  NodeWeight _u_weight;
  EdgeWeight _e_weight;
  NodeWeight _v_weight;
};

// Matcher that checks whether a graph contains an edge identified by the
// weights of its endpoints This helps to test the structure of a graph without
// relying on the order of its nodes or edges, as long as all nodes have an
// unique weight
inline Matcher<const Graph &>
HasEdgeWithWeightedEndpoints(const NodeWeight u_weight, const NodeWeight v_weight) {
  return MakeMatcher(new HasWeightedEdgeWithWeightedEndpointsMatcher(u_weight, 0, v_weight));
}

inline Matcher<const Graph &> HasWeightedEdgeWithWeightedEndpoints(
    const NodeWeight u_weight, const EdgeWeight e_weight, const NodeWeight v_weight
) {
  return MakeMatcher(new HasWeightedEdgeWithWeightedEndpointsMatcher(u_weight, e_weight, v_weight));
}
} // namespace kaminpar::shm::testing
