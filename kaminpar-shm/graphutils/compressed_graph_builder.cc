/*******************************************************************************
 * Builder for compressed graphs.
 *
 * @file:   compressed_graph_builder.cc
 * @author: Daniel Salwasser
 * @date:   03.05.2024
 ******************************************************************************/
#include "kaminpar-shm/graphutils/compressed_graph_builder.h"

#include "kassert/kassert.hpp"

#include "kaminpar-common/parallel/algorithm.h"

namespace kaminpar::shm {

CompressedGraphBuilder::CompressedGraphBuilder(
    const NodeID num_nodes,
    const EdgeID num_edges,
    const bool has_node_weights,
    const bool has_edge_weights,
    const bool sorted
)
    : _num_nodes(num_nodes),
      _num_edges(num_edges),
      _has_node_weights(has_node_weights),
      _has_edge_weights(has_edge_weights),
      _sorted(sorted),
      _cur_node(0),
      _cur_edge(0),
      _compressed_neighborhoods_builder(num_nodes, num_edges, has_edge_weights),
      _total_node_weight(0) {
  if (has_node_weights) {
    _node_weights.resize(num_nodes, 1, static_array::seq);
  }
}

void CompressedGraphBuilder::add_node(std::span<NodeID> neighbors) {
  KASSERT(_cur_node < _num_nodes, "Node ID out of bounds");
  KASSERT((_cur_edge += neighbors.size()) <= _num_edges, "Too many edges added");

  _compressed_neighborhoods_builder.add(_cur_node++, neighbors);
}

void CompressedGraphBuilder::add_node(std::span<std::pair<NodeID, EdgeWeight>> neighborhood) {
  KASSERT(_cur_node < _num_nodes, "Node ID out of bounds");
  KASSERT((_cur_edge += neighborhood.size()) <= _num_edges, "Too many edges added");

  _compressed_neighborhoods_builder.add(_cur_node, neighborhood);
  _cur_node++;
}

void CompressedGraphBuilder::add_node(
    std::span<NodeID> neighbors, std::span<EdgeWeight> edge_weights
) {
  KASSERT(_cur_node < _num_nodes, "Node ID out of bounds");
  KASSERT((_cur_edge += neighbors.size()) <= _num_edges, "Too many edges added");
  KASSERT(neighbors.size() == edge_weights.size(), "Unequal number of neighbors and edge weights");

  if (!_has_edge_weights || edge_weights.empty()) {
    _compressed_neighborhoods_builder.add(_cur_node, neighbors);
  } else {
    const std::size_t num_neighbors = neighbors.size();
    if (_neighborhood.size() < num_neighbors) {
      _neighborhood.resize(num_neighbors);
    }

    for (std::size_t i = 0; i < num_neighbors; ++i) {
      _neighborhood[i] = std::make_pair(neighbors[i], edge_weights[i]);
    }

    _compressed_neighborhoods_builder.add(
        _cur_node, std::span<std::pair<NodeID, EdgeWeight>>(_neighborhood)
    );
  }

  _cur_node++;
}

void CompressedGraphBuilder::add_node_weight(const NodeID node, const NodeWeight weight) {
  KASSERT(_has_node_weights, "Node weights are not stored");
  KASSERT(node < _num_nodes, "Node ID out of bounds");
  KASSERT(weight > 0, "Node weight must be positive");

  _total_node_weight += weight;
  _node_weights[node] = weight;
}

CompressedGraph CompressedGraphBuilder::build() {
  KASSERT(_cur_node == _num_nodes, "Not all nodes have been added");
  KASSERT(_cur_edge == _num_edges, "Not all edges have been added");

  const bool unit_node_weights = std::cmp_equal(_total_node_weight, _num_nodes);
  if (unit_node_weights) {
    _node_weights.free();
  }

  return CompressedGraph(
      _compressed_neighborhoods_builder.build(), std::move(_node_weights), _sorted
  );
}

CompressedGraph compress(const CSRGraph &graph) {
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

CompressedGraph compress(
    std::span<EdgeID> nodes,
    std::span<NodeID> edges,
    std::span<NodeWeight> node_weights,
    std::span<NodeWeight> edge_weights,
    bool sorted
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

ParallelCompressedGraphBuilder::ParallelCompressedGraphBuilder(
    const NodeID num_nodes,
    const EdgeID num_edges,
    const bool has_node_weights,
    const bool has_edge_weights,
    const bool sorted
)
    : _num_nodes(num_nodes),
      _num_edges(num_edges),
      _has_node_weights(has_node_weights),
      _has_edge_weights(has_edge_weights),
      _sorted(sorted),
      _computed_offsets(false),
      _offsets(num_nodes + 1, static_array::noinit),
      _builder(num_nodes, num_edges, has_edge_weights),
      _edges_builder_ets([=] {
        return CompressedEdgesBuilder(
            CompressedEdgesBuilder::num_edges_tag, num_nodes, num_edges, has_edge_weights
        );
      }) {
  if (has_node_weights) {
    _node_weights.resize(num_nodes, static_array::noinit);
  }
}

void ParallelCompressedGraphBuilder::register_neighborhood(
    const NodeID node, std::span<NodeID> neighbors
) {
  KASSERT(node < _num_nodes, "Node ID out of bounds");
  KASSERT(!_computed_offsets, "Offsets have already been computed");

  auto &edges_builder = _edges_builder_ets.local();
  edges_builder.reset();

  edges_builder.add(node, neighbors);
  _offsets[node + 1] = edges_builder.size();
}

void ParallelCompressedGraphBuilder::register_neighborhood(
    const NodeID node, std::span<std::pair<NodeID, EdgeWeight>> neighborhood
) {
  KASSERT(node < _num_nodes, "Node ID out of bounds");
  KASSERT(!_computed_offsets, "Offsets have already been computed");

  auto &edges_builder = _edges_builder_ets.local();
  edges_builder.reset();

  edges_builder.add(node, neighborhood);
  _offsets[node + 1] = edges_builder.size();
}

void ParallelCompressedGraphBuilder::register_neighborhood(
    const NodeID node, std::span<NodeID> neighbors, std::span<EdgeWeight> edge_weights
) {
  KASSERT(node < _num_nodes, "Node ID out of bounds");
  KASSERT(!_computed_offsets, "Offsets have already been computed");
  KASSERT(neighbors.size() == edge_weights.size(), "Unequal number of neighbors and edge weights");

  auto &edges_builder = _edges_builder_ets.local();
  edges_builder.reset();

  if (!_has_edge_weights || edge_weights.empty()) {
    edges_builder.add(node, neighbors);
  } else {
    auto &neighborhood = _neighborhood_ets.local();

    const std::size_t num_neighbors = neighbors.size();
    if (neighborhood.size() < num_neighbors) {
      neighborhood.resize(num_neighbors);
    }

    for (std::size_t i = 0; i < num_neighbors; ++i) {
      neighborhood[i] = std::make_pair(neighbors[i], edge_weights[i]);
    }

    edges_builder.add(node, std::span<std::pair<NodeID, EdgeWeight>>(neighborhood));
  }

  _offsets[node + 1] = edges_builder.size();
}

void ParallelCompressedGraphBuilder::register_neighborhoods(
    const NodeID node,
    std::span<EdgeID> nodes,
    std::span<std::pair<NodeID, EdgeWeight>> neighborhoods
) {
  KASSERT(node < _num_nodes, "Node ID out of bounds");
  KASSERT(!_computed_offsets, "Offsets have already been computed");

  auto &edges_builder = _edges_builder_ets.local();

  const std::size_t num_nodes = nodes.size();
  for (std::size_t i = 0; i < num_nodes; ++i) {
    const EdgeID begin = nodes[i];
    const EdgeID end = (i + 1 == num_nodes) ? neighborhoods.size() : nodes[i + 1];
    const EdgeID length = end - begin;
    auto neighborhood = neighborhoods.subspan(begin, length);

    const NodeID cur_node = node + i;
    edges_builder.reset();

    edges_builder.add(cur_node, neighborhood);
    _offsets[cur_node + 1] = edges_builder.size();
  }
}

void ParallelCompressedGraphBuilder::compute_offsets() {
  KASSERT(!_computed_offsets, "Offsets have already been computed");
  _computed_offsets = true;

  _offsets[0] = 0;
  parallel::prefix_sum(_offsets.begin(), _offsets.end(), _offsets.begin());

  tbb::parallel_for<NodeID>(0, _num_nodes, [&](const NodeID node) {
    _builder.add_node(node, _offsets[node]);
  });
}

void ParallelCompressedGraphBuilder::add_neighborhood(
    const NodeID node, std::span<NodeID> neighbors
) {
  KASSERT(node < _num_nodes, "Node ID out of bounds");
  KASSERT(_computed_offsets, "Offsets have not been computed");

  auto &edges_builder = _edges_builder_ets.local();

  edges_builder.reset();
  edges_builder.add(node, neighbors);

  _builder.add_compressed_edges(
      _offsets[node], edges_builder.size(), edges_builder.compressed_data()
  );
  _builder.record_local_statistics(
      edges_builder.max_degree(),
      edges_builder.total_edge_weight(),
      edges_builder.num_high_degree_nodes(),
      edges_builder.num_high_degree_parts(),
      edges_builder.num_interval_nodes(),
      edges_builder.num_intervals()
  );
}

void ParallelCompressedGraphBuilder::add_neighborhood(
    const NodeID node, std::span<std::pair<NodeID, EdgeWeight>> neighborhood
) {
  KASSERT(node < _num_nodes, "Node ID out of bounds");
  KASSERT(_computed_offsets, "Offsets have not been computed");

  auto &edges_builder = _edges_builder_ets.local();

  edges_builder.reset();
  edges_builder.add(node, neighborhood);

  _builder.add_compressed_edges(
      _offsets[node], edges_builder.size(), edges_builder.compressed_data()
  );
  _builder.record_local_statistics(
      edges_builder.max_degree(),
      edges_builder.total_edge_weight(),
      edges_builder.num_high_degree_nodes(),
      edges_builder.num_high_degree_parts(),
      edges_builder.num_interval_nodes(),
      edges_builder.num_intervals()
  );
}

void ParallelCompressedGraphBuilder::add_neighborhood(
    const NodeID node, std::span<NodeID> neighbors, std::span<EdgeWeight> edge_weights
) {
  KASSERT(node < _num_nodes, "Node ID out of bounds");
  KASSERT(_computed_offsets, "Offsets have not been computed");
  KASSERT(neighbors.size() == edge_weights.size(), "Unequal number of neighbors and edge weights");

  auto &edges_builder = _edges_builder_ets.local();
  edges_builder.reset();

  if (!_has_edge_weights || edge_weights.empty()) {
    edges_builder.add(node, neighbors);
  } else {
    auto &neighborhood = _neighborhood_ets.local();

    const std::size_t num_neighbors = neighbors.size();
    if (neighborhood.size() < num_neighbors) {
      neighborhood.resize(num_neighbors);
    }

    for (std::size_t i = 0; i < num_neighbors; ++i) {
      neighborhood[i] = std::make_pair(neighbors[i], edge_weights[i]);
    }

    edges_builder.add(node, std::span<std::pair<NodeID, EdgeWeight>>(neighborhood));
  }

  _builder.add_compressed_edges(
      _offsets[node], edges_builder.size(), edges_builder.compressed_data()
  );
  _builder.record_local_statistics(
      edges_builder.max_degree(),
      edges_builder.total_edge_weight(),
      edges_builder.num_high_degree_nodes(),
      edges_builder.num_high_degree_parts(),
      edges_builder.num_interval_nodes(),
      edges_builder.num_intervals()
  );
}

void ParallelCompressedGraphBuilder::add_neighborhoods(
    const NodeID node,
    std::span<EdgeID> nodes,
    std::span<std::pair<NodeID, EdgeWeight>> neighborhoods
) {
  KASSERT(node < _num_nodes, "Node ID out of bounds");
  KASSERT(_computed_offsets, "Offsets have not been computed");

  auto &edges_builder = _edges_builder_ets.local();
  edges_builder.reset();

  const std::size_t num_nodes = nodes.size();
  for (std::size_t i = 0; i < num_nodes; ++i) {
    const EdgeID begin = nodes[i];
    const EdgeID end = (i + 1 == num_nodes) ? neighborhoods.size() : nodes[i + 1];
    const EdgeID length = end - begin;
    auto neighborhood = neighborhoods.subspan(begin, length);

    const NodeID cur_node = node + i;
    edges_builder.add(cur_node, neighborhood);
  }

  _builder.add_compressed_edges(
      _offsets[node], edges_builder.size(), edges_builder.compressed_data()
  );
  _builder.record_local_statistics(
      edges_builder.max_degree(),
      edges_builder.total_edge_weight(),
      edges_builder.num_high_degree_nodes(),
      edges_builder.num_high_degree_parts(),
      edges_builder.num_interval_nodes(),
      edges_builder.num_intervals()
  );
}

void ParallelCompressedGraphBuilder::add_neighborhoods(
    const NodeID node,
    std::span<EdgeID> nodes,
    std::span<NodeID> neighbors,
    std::span<EdgeWeight> edge_weights
) {
  KASSERT(node < _num_nodes, "Node ID out of bounds");
  KASSERT(_computed_offsets, "Offsets have not been computed");
  KASSERT(neighbors.size() == edge_weights.size(), "Unequal number of neighbors and edge weights");

  auto &neighborhood = _neighborhood_ets.local();
  auto &edges_builder = _edges_builder_ets.local();
  edges_builder.reset();

  const std::size_t num_nodes = nodes.size();
  for (std::size_t i = 0; i < num_nodes; ++i) {
    const EdgeID begin = nodes[i];
    const EdgeID end = (i + 1 == num_nodes) ? neighbors.size() : nodes[i + 1];
    const EdgeID length = end - begin;

    auto local_neighbors = neighbors.subspan(begin, length);
    auto local_edge_weights = edge_weights.subspan(begin, length);

    const NodeID cur_node = node + i;
    if (!_has_edge_weights || edge_weights.empty()) {
      edges_builder.add(cur_node, local_neighbors);
    } else {
      if (neighborhood.size() < length) {
        neighborhood.resize(length);
      }

      for (std::size_t i = 0; i < length; ++i) {
        neighborhood[i] = std::make_pair(local_neighbors[i], local_edge_weights[i]);
      }

      edges_builder.add(cur_node, std::span<std::pair<NodeID, EdgeWeight>>(neighborhood));
    }
  }

  _builder.add_compressed_edges(
      _offsets[node], edges_builder.size(), edges_builder.compressed_data()
  );
  _builder.record_local_statistics(
      edges_builder.max_degree(),
      edges_builder.total_edge_weight(),
      edges_builder.num_high_degree_nodes(),
      edges_builder.num_high_degree_parts(),
      edges_builder.num_interval_nodes(),
      edges_builder.num_intervals()
  );
}

void ParallelCompressedGraphBuilder::add_node_weight(const NodeID node, const NodeWeight weight) {
  KASSERT(_has_node_weights, "Node weights are not stored");
  KASSERT(node < _num_nodes, "Node ID out of bounds");
  KASSERT(weight > 0, "Node weight must be positive");

  _node_weights[node] = weight;
}

CompressedGraph ParallelCompressedGraphBuilder::build() {
  return CompressedGraph(_builder.build(), std::move(_node_weights), _sorted);
}

CompressedGraph parallel_compress(const CSRGraph &graph) {
  StaticArray<NodeWeight> node_weights;
  if (graph.is_node_weighted()) {
    node_weights.resize(graph.n(), static_array::noinit);

    tbb::parallel_for<NodeID>(0, graph.n(), [&](const NodeID u) {
      node_weights[u] = graph.node_weight(u);
    });
  }

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
        std::move(node_weights),
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
        std::move(node_weights),
        graph.sorted()
    );
  }
}

[[nodiscard]] CompressedGraph parallel_compress(
    std::span<EdgeID> nodes,
    std::span<NodeID> edges,
    std::span<NodeWeight> node_weights,
    std::span<NodeWeight> edge_weights,
    bool sorted
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

} // namespace kaminpar::shm
