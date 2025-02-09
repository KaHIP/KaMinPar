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
#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/concurrent_circular_vector.h"
#include "kaminpar-common/datastructures/maxsize_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/graph_compression/compressed_neighborhoods_builder.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

namespace {

template <bool kHasEdgeWeights, typename DegreeFetcher, typename NeighborhoodFetcher>
[[nodiscard]] CompressedGraph compress_graph(
    const NodeID num_nodes,
    const EdgeID num_edges,
    DegreeFetcher &&fetch_degree,
    NeighborhoodFetcher &&fetch_neighborhood,
    StaticArray<NodeWeight> node_weights = {},
    const bool sorted = false
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

  return CompressedGraph(builder.build(), std::move(node_weights), sorted);
}

} // namespace

[[nodiscard]] CompressedGraph compress(const CSRGraph &graph);

[[nodiscard]] CompressedGraph parallel_compress(const CSRGraph &graph);

template <typename DegreeFetcher, typename NeighborhoodFetcher>
[[nodiscard]] CompressedGraph parallel_compress(
    const NodeID num_nodes,
    const EdgeID num_edges,
    DegreeFetcher &&fetch_degree,
    NeighborhoodFetcher &&fetch_neighborhood,
    StaticArray<NodeWeight> node_weights,
    const bool sorted
) {
  constexpr bool kHasEdgeWeights = false;
  return compress_graph<kHasEdgeWeights, DegreeFetcher, NeighborhoodFetcher>(
      num_nodes,
      num_edges,
      std::forward<DegreeFetcher>(fetch_degree),
      std::forward<NeighborhoodFetcher>(fetch_neighborhood),
      std::move(node_weights),
      sorted
  );
}

template <typename DegreeFetcher, typename NeighborhoodFetcher>
[[nodiscard]] CompressedGraph parallel_compress_weighted(
    const NodeID num_nodes,
    const EdgeID num_edges,
    DegreeFetcher &&fetch_degree,
    NeighborhoodFetcher &&fetch_neighborhood,
    StaticArray<NodeWeight> node_weights,
    const bool sorted
) {
  constexpr bool kHasEdgeWeights = true;
  return compress_graph<kHasEdgeWeights, DegreeFetcher, NeighborhoodFetcher>(
      num_nodes,
      num_edges,
      std::forward<DegreeFetcher>(fetch_degree),
      std::forward<NeighborhoodFetcher>(fetch_neighborhood),
      std::move(node_weights),
      sorted
  );
}

} // namespace kaminpar::shm
