#pragma once

#include "../simple_graph.h"
#include "definitions.h"
#include "graph_converter.h"

#include <stack>

namespace kaminpar::tool::converter {
class StripNodeWeightsProcessor : public GraphProcessor {
public:
  void process(SimpleGraph &graph) override { graph.node_weights.clear(); }
  [[nodiscard]] std::string description() const override { return "Removes all node weights from the graph"; }
};

class StripEdgeWeightsProcessor : public GraphProcessor {
public:
  void process(SimpleGraph &graph) override { graph.edge_weights.clear(); }
  [[nodiscard]] std::string description() const override { return "Removes all edge weights from the graph"; }
};

class StripIsolatedNodesProcessor : public GraphProcessor {
public:
  void process(SimpleGraph &graph) override {
    std::vector<NodeID> remap(graph.n());
    NodeID next_id{0};
    for (const NodeID u : graph.nodes_iter()) {
      if (graph.degree(u) == 0) { continue; }
      remap[u] = next_id;
      graph.nodes[next_id] = graph.nodes[u];
      if (graph.has_node_weights()) { graph.node_weights[next_id] = graph.node_weights[u]; }
      ++next_id;
    }
    graph.nodes[next_id] = graph.m();

    graph.nodes.resize(next_id + 1);
    graph.nodes.shrink_to_fit();
    if (graph.has_node_weights()) {
      graph.node_weights.resize(next_id);
      graph.node_weights.shrink_to_fit();
    }

    for (const EdgeID e : graph.edges_iter()) { graph.edges[e] = remap[graph.edges[e]]; }
  }

  [[nodiscard]] std::string description() const override { return "Removes all nodes with degree 0 from the graph"; }
};

enum class ExtractMetric { N, M };

template<ExtractMetric metric>
class ExtractLargestComponent : public GraphProcessor {
public:
  void process(SimpleGraph &graph) override {
    // find all connected components
    std::vector<bool> visited(graph.n());
    std::vector<NodeID> component(graph.n());
    std::vector<NodeID> component_n;
    std::vector<EdgeID> component_m;
    NodeID current_component{0};
    NodeID num_visited{0};
    NodeID next_node{0};

    std::stack<NodeID> todo;

    while (num_visited < graph.n()) {
      while (visited[next_node]) { ++next_node; }

      todo.push(next_node);
      visited[next_node] = true;
      component_n.push_back(0);
      component_m.push_back(0);

      while (!todo.empty()) {
        const NodeID u = todo.top();
        todo.pop();
        component[u] = current_component;
        ++component_n[current_component];
        ++num_visited;

        for (const auto [e, v] : graph.neighbors_iter(u)) {
          ++component_m[current_component];
          if (!visited[v]) {
            todo.push(v);
            visited[v] = true;
          }
        }
      }

      ++current_component;
    }

    // find the one that we want to keep
    NodeID selected_component{0};

    static_assert(metric == ExtractMetric::N || metric == ExtractMetric::M);
    switch (metric) {
      case ExtractMetric::N:
        selected_component = std::max_element(component_n.begin(), component_n.end()) - component_n.begin();
        break;

      case ExtractMetric::M:
        selected_component = std::max_element(component_m.begin(), component_m.end()) - component_m.begin();
        break;
    }

    LOG << "Extracting component with n'=" << component_n[selected_component]
        << " and m'=" << component_m[selected_component];

    // extract cc
    std::vector<NodeID> remap(graph.n());
    NodeID next_node_id{0};
    for (const NodeID u : graph.nodes_iter()) {
      if (component[u] == selected_component) { remap[u] = next_node_id++; }
    }

    std::vector<EdgeID> extracted_nodes;
    std::vector<NodeID> extracted_edges;
    std::vector<NodeWeight> extracted_node_weights;
    std::vector<EdgeWeight> extracted_edge_weights;
    extracted_nodes.push_back(0);

    for (const NodeID u : graph.nodes_iter()) {
      if (component[u] != selected_component) { continue; }
      for (const auto [e, v] : graph.neighbors_iter(u)) {
        extracted_edges.push_back(remap[v]);
        if (graph.has_edge_weights()) { extracted_edge_weights.push_back(graph.edge_weights[e]); }
      }
      if (graph.has_node_weights()) { extracted_node_weights.push_back(graph.node_weights[u]); }
      extracted_nodes.push_back(extracted_edges.size());
    }

    graph.nodes = std::move(extracted_nodes);
    graph.edges = std::move(extracted_edges);
    graph.node_weights = std::move(extracted_node_weights);
    graph.edge_weights = std::move(extracted_edge_weights);
  }

  [[nodiscard]] std::string description() const override {
    switch (metric) {
      case ExtractMetric::N:
        return "Extracts the connected component with the largest number of nodes and discards the rest of the graph";

      case ExtractMetric::M:
        return "Extracts the connected component with the largest number of edges and discards the rest of the graph";
    }

    __builtin_unreachable();
  }
};
} // namespace kaminpar::tool::converter
