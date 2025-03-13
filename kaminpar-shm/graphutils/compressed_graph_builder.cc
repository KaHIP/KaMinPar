/*******************************************************************************
 * Builder for compressed graphs.
 *
 * @file:   compressed_graph_builder.cc
 * @author: Daniel Salwasser
 * @date:   03.05.2024
 ******************************************************************************/
#include "kaminpar-shm/graphutils/compressed_graph_builder.h"

#include "kaminpar-shm/datastructures/csr_graph.h"

namespace kaminpar::shm {

Graph compress(const CSRGraph &graph) {
  const bool store_node_weights = graph.is_node_weighted();
  const bool store_edge_weights = graph.is_edge_weighted();

  CompressedGraphBuilder builder(
      graph.n(), graph.m(), store_node_weights, store_edge_weights, graph.sorted()
  );

  std::vector<std::pair<NodeID, EdgeWeight>> neighbourhood;
  neighbourhood.reserve(graph.max_degree());

  for (const NodeID u : graph.nodes()) {
    graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      neighbourhood.emplace_back(v, w);
    });

    builder.add_node(neighbourhood);
    if (store_node_weights) {
      builder.add_node_weight(u, graph.node_weight(u));
    }

    neighbourhood.clear();
  }

  return builder.build();
}

Graph parallel_compress(const CSRGraph &graph) {
  const auto fetch_degree = [&](const NodeID u) {
    return graph.degree(u);
  };

  if (graph.is_edge_weighted()) {
    using Edge = std::pair<NodeID, EdgeWeight>;
    const auto fetch_neighborhood = [&](const NodeID u, std::span<Edge> neighborhood) {
      NodeID i = 0;
      graph.adjacent_nodes(u, [&](const NodeID u, const EdgeWeight w) {
        neighborhood[i++] = std::make_pair(u, w);
      });
    };

    return parallel_compress_weighted(
        graph.n(),
        graph.m(),
        fetch_degree,
        fetch_neighborhood,
        graph.raw_node_weights(),
        graph.sorted()
    );
  } else {
    const auto fetch_neighborhood = [&](const NodeID u, std::span<NodeID> neighborhood) {
      // Fill the provided neighborhood array via memcpy since we know the layout of the graph to
      // compress. Alternatively, copy the neighbors one-by-one as follows:
      //
      // NodeID i = 0;
      // graph.adjacent_nodes(u, [&](const NodeID u) { neighborhood[i++] = u; });
      //

      const NodeID *raw_edges = graph.raw_edges().data();
      std::memcpy(
          neighborhood.data(), raw_edges + graph.first_edge(u), graph.degree(u) * sizeof(NodeID)
      );
    };

    return parallel_compress(
        graph.n(),
        graph.m(),
        fetch_degree,
        fetch_neighborhood,
        graph.raw_node_weights(),
        graph.sorted()
    );
  }
}

Graph compress(
    std::span<const EdgeID> nodes,
    std::span<const NodeID> edges,
    std::span<const NodeWeight> node_weights,
    std::span<const NodeWeight> edge_weights,
    const bool sorted
) {
  const NodeID n = nodes.size() - 1;
  const EdgeID m = edges.size();

  const bool store_node_weights = !node_weights.empty();
  const bool store_edge_weights = !edge_weights.empty();

  CompressedGraphBuilder builder(n, m, store_node_weights, store_edge_weights, sorted);

  std::vector<std::pair<NodeID, EdgeWeight>> neighbourhood;
  for (NodeID u = 0; u < n; ++u) {
    const EdgeID begin = nodes[u];
    const EdgeID end = nodes[u + 1];

    for (EdgeID e = begin; e < end; ++e) {
      const NodeID v = edges[e];
      const EdgeWeight w = store_edge_weights ? edge_weights[e] : 1;
      neighbourhood.emplace_back(v, w);
    }

    builder.add_node(neighbourhood);
    if (store_node_weights) {
      builder.add_node_weight(u, node_weights[u]);
    }

    neighbourhood.clear();
  }

  return builder.build();
}

Graph parallel_compress(
    std::span<const EdgeID> nodes,
    std::span<const NodeID> edges,
    std::span<const NodeWeight> node_weights,
    std::span<const NodeWeight> edge_weights,
    const bool sorted
) {
  const NodeID n = nodes.size() - 1;
  const EdgeID m = edges.size();

  const auto fetch_degree = [&](const NodeID u) {
    return nodes[u + 1] - nodes[u];
  };

  const bool store_edge_weights = !edge_weights.empty();
  if (store_edge_weights) {
    using Edge = std::pair<NodeID, EdgeWeight>;
    const auto fetch_neighborhood = [&](const NodeID u, std::span<Edge> neighborhood) {
      NodeID i = 0;

      const EdgeID begin = nodes[u];
      const EdgeID end = nodes[u + 1];
      for (EdgeID e = begin; e < end; ++e) {
        neighborhood[i++] = {edges[e], edge_weights[e]};
      }
    };

    return parallel_compress_weighted(
        n, m, fetch_degree, fetch_neighborhood, {node_weights.begin(), node_weights.end()}, sorted
    );
  } else {
    const auto fetch_neighborhood = [&](const NodeID u, std::span<NodeID> neighborhood) {
      const EdgeID begin = nodes[u];
      const EdgeID end = nodes[u + 1];
      std::memcpy(neighborhood.data(), edges.data() + begin, (end - begin) * sizeof(NodeID));
    };

    return parallel_compress(
        n, m, fetch_degree, fetch_neighborhood, {node_weights.begin(), node_weights.end()}, sorted
    );
  }
}

CompressedGraphBuilder::CompressedGraphBuilder(
    const NodeID num_nodes,
    const EdgeID num_edges,
    const bool has_node_weights,
    const bool has_edge_weights,
    const bool sorted
)
    : _impl(std::make_unique<Impl>(num_nodes, num_edges, has_node_weights, has_edge_weights, sorted)
      ) {}

void CompressedGraphBuilder::add_node(std::span<NodeID> neighbors) {
  _impl->add_node(neighbors);
}

void CompressedGraphBuilder::add_node(std::span<std::pair<NodeID, EdgeWeight>> neighborhood) {
  _impl->add_node(neighborhood);
}

void CompressedGraphBuilder::add_node(
    std::span<NodeID> neighbors, std::span<EdgeWeight> edge_weights
) {
  _impl->add_node(neighbors, edge_weights);
}

void CompressedGraphBuilder::add_node_weight(const NodeID node, const NodeWeight weight) {
  _impl->add_node_weight(node, weight);
}

Graph CompressedGraphBuilder::build() {
  return _impl->build();
}

ParallelCompressedGraphBuilder::ParallelCompressedGraphBuilder(
    const NodeID num_nodes,
    const EdgeID num_edges,
    const bool has_node_weights,
    const bool has_edge_weights,
    const bool sorted
)
    : _impl(std::make_unique<Impl>(num_nodes, num_edges, has_node_weights, has_edge_weights, sorted)
      ) {}

void ParallelCompressedGraphBuilder::register_neighborhood(
    const NodeID node, std::span<NodeID> neighbors
) {
  _impl->register_neighborhood(node, neighbors);
}

void ParallelCompressedGraphBuilder::register_neighborhood(
    const NodeID node, std::span<std::pair<NodeID, EdgeWeight>> neighborhood
) {
  _impl->register_neighborhood(node, neighborhood);
}

void ParallelCompressedGraphBuilder::register_neighborhood(
    const NodeID node, std::span<NodeID> neighbors, std::span<EdgeWeight> edge_weights
) {
  _impl->register_neighborhood(node, neighbors, edge_weights);
}

void ParallelCompressedGraphBuilder::register_neighborhoods(
    const NodeID node,
    std::span<EdgeID> nodes,
    std::span<std::pair<NodeID, EdgeWeight>> neighborhoods
) {
  _impl->register_neighborhoods(node, nodes, neighborhoods);
}

void ParallelCompressedGraphBuilder::compute_offsets() {
  _impl->compute_offsets();
}

void ParallelCompressedGraphBuilder::add_neighborhood(
    const NodeID node, std::span<NodeID> neighbors
) {
  _impl->add_neighborhood(node, neighbors);
}

void ParallelCompressedGraphBuilder::add_neighborhood(
    const NodeID node, std::span<std::pair<NodeID, EdgeWeight>> neighborhood
) {
  _impl->add_neighborhood(node, neighborhood);
}

void ParallelCompressedGraphBuilder::add_neighborhood(
    const NodeID node, std::span<NodeID> neighbors, std::span<EdgeWeight> edge_weights
) {
  _impl->add_neighborhood(node, neighbors, edge_weights);
}

void ParallelCompressedGraphBuilder::add_neighborhoods(
    const NodeID node,
    std::span<EdgeID> nodes,
    std::span<std::pair<NodeID, EdgeWeight>> neighborhoods
) {
  _impl->add_neighborhoods(node, nodes, neighborhoods);
}

void ParallelCompressedGraphBuilder::add_neighborhoods(
    const NodeID node,
    std::span<EdgeID> nodes,
    std::span<NodeID> neighbors,
    std::span<EdgeWeight> edge_weights
) {
  _impl->add_neighborhoods(node, nodes, neighbors, edge_weights);
}

void ParallelCompressedGraphBuilder::add_node_weight(const NodeID node, const NodeWeight weight) {
  _impl->add_node_weight(node, weight);
}

Graph ParallelCompressedGraphBuilder::build() {
  return _impl->build();
}

} // namespace kaminpar::shm
