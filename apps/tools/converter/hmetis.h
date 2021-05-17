#pragma once

#include "definitions.h"
#include "graph_converter.h"

#include <fstream>
#include <memory>

namespace kaminpar::tool::converter {
class HMetisWriter : public GraphWriter {
public:
  void write(const std::string &filename, SimpleGraph graph, const std::string &) override {
    const bool write_node_weights = !graph.node_weights.empty();
    const bool write_edge_weights = !graph.edge_weights.empty();

    std::ofstream out(filename);
    if (!out) { FATAL_ERROR << "Cannot write to " << filename; }

    // header
    out << graph.edges.size() / 2 << " " << graph.nodes.size() - 1;
    if (write_node_weights || write_edge_weights) { out << " " << write_node_weights << write_edge_weights; }
    out << "\n";

    // hyperedge for each undirected edge
    for (NodeID u = 0; u + 1 < graph.nodes.size(); ++u) {
      for (EdgeID e = graph.nodes[u]; e < graph.nodes[u + 1]; ++e) {
        const NodeID v = graph.edges[e];
        if (u < v) {
          if (write_edge_weights) { out << graph.edge_weights[e] << " "; }
          out << (u + 1) << " " << (v + 1) << "\n";
        }
      }
    }

    // hypernode weights
    if (write_node_weights) {
      for (NodeID u = 0; u + 1 < graph.nodes.size(); ++u) { out << graph.node_weights[u] << "\n"; }
    }
  }

  [[nodiscard]] std::string description() const override { return "hMETIS hypergraph format"; }
};
} // namespace kaminpar::tool::converter
