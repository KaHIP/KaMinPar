/*******************************************************************************
 * Builders for compressed graphs.
 *
 * @file:   compressed_graph_builder.h
 * @author: Daniel Salwasser
 * @date:   03.05.2024
 ******************************************************************************/
#pragma once

#include <span>
#include <utility>
#include <vector>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/datastructures/compressed_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/concurrent_circular_vector.h"
#include "kaminpar-common/datastructures/maxsize_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/graph_compression/compressed_neighborhoods_builder.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

namespace {

template <bool kHasEdgeWeights, typename DegreeFetcher, typename NeighborhoodFetcher>
[[nodiscard]] CompressedGraph compress_graph(
    const NodeID num_nodes,
    const EdgeID num_edges,
    DegreeFetcher &&fetch_degree,
    NeighborhoodFetcher &&fetch_neighborhood,
    std::span<const NodeWeight> node_weights,
    const bool sorted
) {
  using NeighbourhoodView = std::
      conditional_t<kHasEdgeWeights, std::span<std::pair<NodeID, EdgeWeight>>, std::span<NodeID>>;
  static_assert(std::is_invocable_r_v<NodeID, DegreeFetcher, NodeID>);
  static_assert(std::is_invocable_v<NeighborhoodFetcher, NodeID, NeighbourhoodView>);
  KASSERT(node_weights.empty() || node_weights.size() == num_nodes);

  // To compress the graph in parallel, its nodes are split into chunks. Each thread fetches a chunk
  // and compresses the neighbourhoods of the nodes in its chunk. The compressed neighborhoods are
  // meanwhile stored in a buffer. They are moved into the compressed edge array when the (total)
  // length of the compressed neighborhoods of the previous chunks is determined.

  // First step: Create the chunks such that each chunk has about the same number of edges.
  // @TODO: Parallelize
  constexpr std::size_t kNumChunks = 5000;
  const EdgeID max_chunk_order = num_edges / kNumChunks;
  std::vector<std::pair<NodeID, NodeID>> chunks;

  NodeID max_degree = 0;
  NodeID max_chunk_size = 0;
  TIMED_SCOPE("Compute chunks") {
    NodeID cur_chunk_start = 0;
    EdgeID cur_chunk_order = 0;

    for (NodeID u = 0; u < num_nodes; ++u) {
      const NodeID degree = fetch_degree(u);

      max_degree = std::max(max_degree, degree);
      cur_chunk_order += degree;

      if (cur_chunk_order >= max_chunk_order) {
        // If there is a node whose neighborhood is larger than the chunk size limit, create a chunk
        // consisting only of this node.
        const bool singleton_chunk = cur_chunk_start == u;
        if (singleton_chunk) {
          chunks.emplace_back(cur_chunk_start, u + 1);
          max_chunk_size = std::max<NodeID>(max_chunk_size, 1);

          cur_chunk_start = u + 1;
          cur_chunk_order = 0;
          continue;
        }

        chunks.emplace_back(cur_chunk_start, u);
        max_chunk_size = std::max<NodeID>(max_chunk_size, u - cur_chunk_start);

        cur_chunk_start = u;
        cur_chunk_order = degree;
      }
    }

    // If the last chunk is smaller than the chunk size limit, add it explicitly.
    if (cur_chunk_start != num_nodes) {
      chunks.emplace_back(cur_chunk_start, num_nodes);
      max_chunk_size = std::max<NodeID>(max_chunk_size, num_nodes - cur_chunk_start);
    }
  };

  // Second step: Initialize the data structures used to build the compressed graph in parallel.
  ParallelCompressedNeighborhoodsBuilder<NodeID, EdgeID, EdgeWeight> builder(
      num_nodes, num_edges, kHasEdgeWeights
  );

  tbb::enumerable_thread_specific<MaxSizeVector<EdgeID>> offsets_ets([&] {
    return MaxSizeVector<EdgeID>(max_chunk_size);
  });

  using Neighbourhood = std::conditional_t<
      kHasEdgeWeights,
      MaxSizeVector<std::pair<NodeID, EdgeWeight>>,
      MaxSizeVector<NodeID>>;
  const std::size_t max_capacity = std::max<std::size_t>(max_chunk_order, max_degree);
  tbb::enumerable_thread_specific<Neighbourhood> neighbourhood_ets([&] {
    return Neighbourhood(max_capacity);
  });

  using CompressedEdgesBuilder = kaminpar::CompressedEdgesBuilder<NodeID, EdgeID, EdgeWeight>;
  tbb::enumerable_thread_specific<CompressedEdgesBuilder> edges_builder_ets([&] {
    return CompressedEdgesBuilder(
        CompressedEdgesBuilder::degree_tag, num_nodes, max_degree, kHasEdgeWeights
    );
  });

  const std::size_t num_threads = tbb::this_task_arena::max_concurrency();
  ConcurrentCircularVectorMutex<NodeID, EdgeID> buffer(num_threads);

  // Third step: Compress the chunks in parallel.
  tbb::parallel_for<NodeID>(0, chunks.size(), [&](const auto) {
    auto &offsets = offsets_ets.local();
    auto &neighbourhood = neighbourhood_ets.local();
    auto &edges_builder = edges_builder_ets.local();
    edges_builder.reset();

    const NodeID chunk = buffer.next();
    const auto [start, end] = chunks[chunk];

    // Compress the neighborhoods of the nodes in the fetched chunk.
    for (NodeID u = start; u < end; ++u) {
      const NodeID degree = fetch_degree(u);
      if (neighbourhood.size() < degree) {
        neighbourhood.resize(degree);
      }

      NeighbourhoodView neighborhood_view(neighbourhood.begin(), degree);
      fetch_neighborhood(u, neighborhood_view);

      const EdgeID local_offset = edges_builder.add(u, neighborhood_view);
      offsets.push_back(local_offset);
      neighbourhood.clear();
    }

    // Wait for the threads that process previous chunks to finish as well.
    const EdgeID compressed_neighborhoods_size = edges_builder.size();
    const EdgeID offset = buffer.fetch_and_update(chunk, compressed_neighborhoods_size);

    // Store the offset into the compressed edge array for each node in the chunk and copy the
    // compressed neighborhoods into the actual compressed edge array.
    for (NodeID u = start; u < end; ++u) {
      const EdgeID local_offset = offsets[u - start];
      builder.add_node(u, offset + local_offset);
    }
    offsets.clear();

    builder.add_compressed_edges(offset, edges_builder.size(), edges_builder.compressed_data());
    builder.record_local_statistics(
        edges_builder.max_degree(),
        edges_builder.total_edge_weight(),
        edges_builder.num_high_degree_nodes(),
        edges_builder.num_high_degree_parts(),
        edges_builder.num_interval_nodes(),
        edges_builder.num_intervals()
    );
  });

  StaticArray<NodeWeight> node_weights_array;
  if (!node_weights.empty()) {
    node_weights_array = StaticArray<NodeWeight>(node_weights.begin(), node_weights.end());
  }

  return CompressedGraph(builder.build(), std::move(node_weights_array), sorted);
}

} // namespace

template <typename DegreeFetcher, typename NeighborhoodFetcher>
[[nodiscard]] CompressedGraph parallel_compress(
    const NodeID num_nodes,
    const EdgeID num_edges,
    DegreeFetcher &&fetch_degree,
    NeighborhoodFetcher &&fetch_neighborhood,
    std::span<const NodeWeight> node_weights,
    const bool sorted
) {
  constexpr bool kHasEdgeWeights = false;
  return compress_graph<kHasEdgeWeights, DegreeFetcher, NeighborhoodFetcher>(
      num_nodes,
      num_edges,
      std::forward<DegreeFetcher>(fetch_degree),
      std::forward<NeighborhoodFetcher>(fetch_neighborhood),
      node_weights,
      sorted
  );
}

template <typename DegreeFetcher, typename NeighborhoodFetcher>
[[nodiscard]] CompressedGraph parallel_compress_weighted(
    const NodeID num_nodes,
    const EdgeID num_edges,
    DegreeFetcher &&fetch_degree,
    NeighborhoodFetcher &&fetch_neighborhood,
    std::span<const NodeWeight> node_weights,
    const bool sorted
) {
  constexpr bool kHasEdgeWeights = true;
  return compress_graph<kHasEdgeWeights, DegreeFetcher, NeighborhoodFetcher>(
      num_nodes,
      num_edges,
      std::forward<DegreeFetcher>(fetch_degree),
      std::forward<NeighborhoodFetcher>(fetch_neighborhood),
      node_weights,
      sorted
  );
}

struct CompressedGraphBuilder::Impl {
  using CompressedNeighborhoodsBuilder =
      kaminpar::CompressedNeighborhoodsBuilder<NodeID, EdgeID, EdgeWeight>;

public:
  Impl(
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

  void add_node(std::span<NodeID> neighbors) {
    KASSERT(_cur_node < _num_nodes, "Node ID out of bounds");
    KASSERT((_cur_edge += neighbors.size()) <= _num_edges, "Too many edges added");

    _compressed_neighborhoods_builder.add(_cur_node++, neighbors);
  }

  void add_node(std::span<std::pair<NodeID, EdgeWeight>> neighborhood) {
    KASSERT(_cur_node < _num_nodes, "Node ID out of bounds");
    KASSERT((_cur_edge += neighborhood.size()) <= _num_edges, "Too many edges added");

    _compressed_neighborhoods_builder.add(_cur_node, neighborhood);
    _cur_node++;
  }

  void add_node(std::span<NodeID> neighbors, std::span<EdgeWeight> edge_weights) {
    KASSERT(_cur_node < _num_nodes, "Node ID out of bounds");
    KASSERT((_cur_edge += neighbors.size()) <= _num_edges, "Too many edges added");
    KASSERT(
        neighbors.size() == edge_weights.size(), "Unequal number of neighbors and edge weights"
    );

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

  void add_node_weight(const NodeID node, const NodeWeight weight) {
    KASSERT(_has_node_weights, "Node weights are not stored");
    KASSERT(node < _num_nodes, "Node ID out of bounds");
    KASSERT(weight > 0, "Node weight must be positive");

    _total_node_weight += weight;
    _node_weights[node] = weight;
  }

  CompressedGraph build() {
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

private:
  NodeID _num_nodes;
  EdgeID _num_edges;
  bool _has_node_weights;
  bool _has_edge_weights;
  bool _sorted;

  NodeID _cur_node;
  EdgeID _cur_edge;
  CompressedNeighborhoodsBuilder _compressed_neighborhoods_builder;
  std::vector<std::pair<NodeID, EdgeWeight>> _neighborhood;
  NodeWeight _total_node_weight;
  StaticArray<NodeWeight> _node_weights;
};

struct ParallelCompressedGraphBuilder::Impl {
  using CompressedEdgesBuilder = kaminpar::CompressedEdgesBuilder<NodeID, EdgeID, EdgeWeight>;
  using ParallelCompressedNeighborhoodsBuilder =
      kaminpar::ParallelCompressedNeighborhoodsBuilder<NodeID, EdgeID, EdgeWeight>;

public:
  Impl(
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

  void register_neighborhood(const NodeID node, std::span<NodeID> neighbors) {
    KASSERT(node < _num_nodes, "Node ID out of bounds");
    KASSERT(!_computed_offsets, "Offsets have already been computed");

    auto &edges_builder = _edges_builder_ets.local();
    edges_builder.reset();

    edges_builder.add(node, neighbors);
    _offsets[node + 1] = edges_builder.size();
  }

  void
  register_neighborhood(const NodeID node, std::span<std::pair<NodeID, EdgeWeight>> neighborhood) {
    KASSERT(node < _num_nodes, "Node ID out of bounds");
    KASSERT(!_computed_offsets, "Offsets have already been computed");

    auto &edges_builder = _edges_builder_ets.local();
    edges_builder.reset();

    edges_builder.add(node, neighborhood);
    _offsets[node + 1] = edges_builder.size();
  }

  void register_neighborhood(
      const NodeID node, std::span<NodeID> neighbors, std::span<EdgeWeight> edge_weights
  ) {
    KASSERT(node < _num_nodes, "Node ID out of bounds");
    KASSERT(!_computed_offsets, "Offsets have already been computed");
    KASSERT(
        neighbors.size() == edge_weights.size(), "Unequal number of neighbors and edge weights"
    );

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

  void register_neighborhoods(
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

  void compute_offsets() {
    KASSERT(!_computed_offsets, "Offsets have already been computed");
    _computed_offsets = true;

    _offsets[0] = 0;
    parallel::prefix_sum(_offsets.begin(), _offsets.end(), _offsets.begin());

    tbb::parallel_for<NodeID>(0, _num_nodes, [&](const NodeID node) {
      _builder.add_node(node, _offsets[node]);
    });
  }

  void add_neighborhood(const NodeID node, std::span<NodeID> neighbors) {
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

  void add_neighborhood(const NodeID node, std::span<std::pair<NodeID, EdgeWeight>> neighborhood) {
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

  void add_neighborhood(
      const NodeID node, std::span<NodeID> neighbors, std::span<EdgeWeight> edge_weights
  ) {
    KASSERT(node < _num_nodes, "Node ID out of bounds");
    KASSERT(_computed_offsets, "Offsets have not been computed");
    KASSERT(
        neighbors.size() == edge_weights.size(), "Unequal number of neighbors and edge weights"
    );

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

  void add_neighborhoods(
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

  void add_neighborhoods(
      const NodeID node,
      std::span<EdgeID> nodes,
      std::span<NodeID> neighbors,
      std::span<EdgeWeight> edge_weights
  ) {
    KASSERT(node < _num_nodes, "Node ID out of bounds");
    KASSERT(_computed_offsets, "Offsets have not been computed");
    KASSERT(
        neighbors.size() == edge_weights.size(), "Unequal number of neighbors and edge weights"
    );

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

  void add_node_weight(const NodeID node, const NodeWeight weight) {
    KASSERT(_has_node_weights, "Node weights are not stored");
    KASSERT(node < _num_nodes, "Node ID out of bounds");
    KASSERT(weight > 0, "Node weight must be positive");

    _node_weights[node] = weight;
  }

  CompressedGraph build() {
    return CompressedGraph(_builder.build(), std::move(_node_weights), _sorted);
  }

private:
  NodeID _num_nodes;
  EdgeID _num_edges;
  bool _has_node_weights;
  bool _has_edge_weights;
  bool _sorted;

  bool _computed_offsets;
  StaticArray<EdgeID> _offsets;
  StaticArray<NodeWeight> _node_weights;
  ParallelCompressedNeighborhoodsBuilder _builder;
  tbb::enumerable_thread_specific<CompressedEdgesBuilder> _edges_builder_ets;
  tbb::enumerable_thread_specific<std::vector<std::pair<NodeID, EdgeWeight>>> _neighborhood_ets;
};

} // namespace kaminpar::shm
