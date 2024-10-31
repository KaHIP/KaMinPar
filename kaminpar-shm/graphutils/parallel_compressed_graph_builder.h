/*******************************************************************************
 * Parallel builder for compressed graphs.
 *
 * @file:   parallel_compressed_graph_builder.h
 * @author: Daniel Salwasser
 * @date:   03.05.2024
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/compressed_graph.h"
#include "kaminpar-shm/datastructures/csr_graph.h"

#include "kaminpar-common/datastructures/concurrent_circular_vector.h"
#include "kaminpar-common/datastructures/maxsize_vector.h"
#include "kaminpar-common/graph_compression/compressed_neighborhoods_builder.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

namespace {
SET_DEBUG(false);

namespace debug {
using Duration = std::chrono::high_resolution_clock::duration;

struct Stats {
  Duration compression_time{0};
  Duration sync_time{0};
  Duration copy_time{0};

  std::size_t num_chunks{0};
  std::size_t num_edges{0};
};

template <typename Lambda> decltype(auto) scoped_time(auto &elapsed, Lambda &&l) {
  constexpr bool kNonReturning = std::is_void_v<std::invoke_result_t<Lambda>>;

  if constexpr (kDebug) {
    if constexpr (kNonReturning) {
      auto start = std::chrono::high_resolution_clock::now();
      l();
      auto end = std::chrono::high_resolution_clock::now();
      elapsed += end - start;
    } else {
      auto start = std::chrono::high_resolution_clock::now();
      decltype(auto) val = l();
      auto end = std::chrono::high_resolution_clock::now();
      elapsed += end - start;
      return val;
    }
  } else {
    return l();
  }
}

void print_graph_compression_stats(const auto &stats_ets) {
  DBG << "Chunk distribution:";

  std::size_t cur_thread = 0;
  for (const auto &stats : stats_ets) {
    DBG << " t" << ++cur_thread << ": " << stats.num_chunks;
  }

  DBG << "Edge distribution:";

  cur_thread = 0;
  for (const auto &stats : stats_ets) {
    DBG << " t" << ++cur_thread << ": " << stats.num_edges;
  }

  DBG << "Time distribution: (compression, sync, copy) [s]";

  const auto to_sec = [&](auto elapsed) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() / 1000.0;
  };

  Duration total_time_compression(0);
  Duration total_time_sync(0);
  Duration total_time_copy(0);

  cur_thread = 0;
  for (const auto &stats : stats_ets) {
    total_time_compression += stats.compression_time;
    total_time_sync += stats.sync_time;
    total_time_copy += stats.copy_time;

    DBG << " t" << ++cur_thread << ": " << to_sec(stats.compression_time) << ' '
        << to_sec(stats.sync_time) << ' ' << to_sec(stats.copy_time);
  }

  DBG << " sum: " << to_sec(total_time_compression) << ' ' << to_sec(total_time_sync) << ' '
      << to_sec(total_time_copy);
}

void print_compressed_graph_stats(const auto &stats_ets) {
  std::size_t _total_adjacent_nodes_num_bytes = 0;
  std::size_t _total_edge_weights_num_bytes = 0;

  for (const auto &neighbourhood_builder : stats_ets) {
    _total_adjacent_nodes_num_bytes += neighbourhood_builder.num_adjacent_node_bytes();
    _total_edge_weights_num_bytes += neighbourhood_builder.num_edge_weights_bytes();
  }

  const auto to_mb = [](const auto num_bytes) {
    return num_bytes / static_cast<float>(1024 * 1024);
  };

  DBG << "Compressed adjacent nodes memory space: " << to_mb(_total_adjacent_nodes_num_bytes)
      << " MiB";
  DBG << "Compressed edge weights memory space: " << to_mb(_total_edge_weights_num_bytes) << " MiB";
}

} // namespace debug

template <
    bool kHasEdgeWeights,
    typename PermutationMapper,
    typename DegreeMapper,
    typename NodeMapper,
    typename EdgeMapper,
    typename NodeWeightMapper,
    typename EdgeWeightMapper>
[[nodiscard]] CompressedGraph compute_compressed_graph(
    const NodeID num_nodes,
    const EdgeID num_edges,
    const bool has_node_weights,
    const bool sorted,
    PermutationMapper &&node_mapper,
    DegreeMapper &&degrees,
    NodeMapper &&nodes,
    EdgeMapper &&edges,
    NodeWeightMapper &&node_weights,
    EdgeWeightMapper &&edge_weights
) {
  // To compress the graph in parallel the nodes are split into chunks. Each parallel task fetches
  // a chunk and compresses the neighbourhoods of the corresponding nodes. The compressed
  // neighborhoods are meanwhile stored in a buffer. They are moved into the compressed edge array
  // when the (total) length of the compressed neighborhoods of the previous chunks is determined.

  // First step: Create the chunks so that each chunk has about the same number of edges.
  constexpr std::size_t kNumChunks = 5000;
  const EdgeID max_chunk_order = num_edges / kNumChunks;
  std::vector<std::tuple<NodeID, NodeID, EdgeID>> chunks;

  NodeID max_degree = 0;
  NodeID max_chunk_size = 0;
  TIMED_SCOPE("Compute chunks") {
    NodeID cur_chunk_start = 0;
    EdgeID cur_chunk_order = 0;
    EdgeID cur_first_edge = 0;
    for (NodeID i = 0; i < num_nodes; ++i) {
      const NodeID node = node_mapper(i);
      const NodeID degree = degrees(node);

      max_degree = std::max(max_degree, degree);
      cur_chunk_order += degree;

      if (cur_chunk_order >= max_chunk_order) {
        // If there is a node whose neighborhood is larger than the chunk size limit, create a chunk
        // consisting only of this node.
        const bool singleton_chunk = cur_chunk_start == i;
        if (singleton_chunk) {
          chunks.emplace_back(cur_chunk_start, i + 1, cur_first_edge);
          max_chunk_size = std::max<NodeID>(max_chunk_size, 1);

          cur_chunk_start = i + 1;
          cur_first_edge += degree;
          cur_chunk_order = 0;
          continue;
        }

        chunks.emplace_back(cur_chunk_start, i, cur_first_edge);
        max_chunk_size = std::max<NodeID>(max_chunk_size, i - cur_chunk_start);

        cur_chunk_start = i;
        cur_first_edge += cur_chunk_order - degree;
        cur_chunk_order = degree;
      }
    }

    // If the last chunk is smaller than the chunk size limit, add it explicitly.
    if (cur_chunk_start != num_nodes) {
      chunks.emplace_back(cur_chunk_start, num_nodes, cur_first_edge);
      max_chunk_size = std::max<NodeID>(max_chunk_size, num_nodes - cur_chunk_start);
    }
  };

  // Second step: Initializes the data structures used to build the compressed graph in parallel.
  ParallelCompressedNeighborhoodsBuilder<NodeID, EdgeID, EdgeWeight> builder(
      num_nodes, num_edges, kHasEdgeWeights
  );

  StaticArray<NodeWeight> node_weights_array;
  if (has_node_weights) {
    node_weights_array.resize(num_nodes, static_array::noinit);
  }

  tbb::enumerable_thread_specific<MaxSizeVector<EdgeID>> offsets_ets([&] {
    return MaxSizeVector<EdgeID>(max_chunk_size);
  });

  using Neighbourhood = std::conditional_t<
      kHasEdgeWeights,
      MaxSizeVector<std::pair<NodeID, EdgeWeight>>,
      MaxSizeVector<NodeID>>;
  tbb::enumerable_thread_specific<Neighbourhood> neighbourhood_ets([&] {
    const std::size_t max_capacity = std::max<std::size_t>(max_chunk_order, max_degree);
    return Neighbourhood(max_capacity);
  });

  using CompressedEdgesBuilder = kaminpar::CompressedEdgesBuilder<NodeID, EdgeID, EdgeWeight>;
  tbb::enumerable_thread_specific<CompressedEdgesBuilder> edges_builder_ets([&] {
    return CompressedEdgesBuilder(
        CompressedEdgesBuilder::degree_tag,
        num_nodes,
        max_degree,
        kHasEdgeWeights,
        builder.edge_weights()
    );
  });

  const std::size_t num_threads = tbb::this_task_arena::max_concurrency();
  ConcurrentCircularVectorMutex<NodeID, EdgeID> buffer(num_threads);

  // Third step: Compress the chunks in parallel.
  tbb::enumerable_thread_specific<debug::Stats> dbg_ets;
  tbb::parallel_for<NodeID>(0, chunks.size(), [&](const auto) {
    auto &dbg = dbg_ets.local();
    IF_DBG dbg.num_chunks++;

    auto &offsets = offsets_ets.local();
    auto &neighbourhood = neighbourhood_ets.local();
    auto &edges_builder = edges_builder_ets.local();

    const NodeID chunk = buffer.next();
    const auto [start, end, first_edge] = chunks[chunk];

    edges_builder.init(first_edge);

    // Compress the neighborhoods of the nodes in the fetched chunk.
    debug::scoped_time(dbg.compression_time, [&] {
      for (NodeID i = start; i < end; ++i) {
        const NodeID node = node_mapper(i);
        const NodeID degree = degrees(node);
        IF_DBG dbg.num_edges += degree;

        EdgeID edge = nodes(node);
        for (NodeID j = 0; j < degree; ++j) {
          const NodeID adjacent_node = edges(edge);

          if constexpr (kHasEdgeWeights) {
            const EdgeWeight edge_weight = edge_weights(edge);
            neighbourhood.emplace_back(adjacent_node, edge_weight);
          } else {
            neighbourhood.push_back(adjacent_node);
          }

          edge += 1;
        }

        const EdgeID local_offset = edges_builder.add(i, neighbourhood);
        offsets.push_back(local_offset);

        neighbourhood.clear();
      }
    });

    // Wait for the parallel tasks that process the previous chunks to finish.
    const EdgeID offset = debug::scoped_time(dbg.sync_time, [&] {
      const EdgeID compressed_neighborhoods_size = edges_builder.size();
      return buffer.fetch_and_update(chunk, compressed_neighborhoods_size);
    });

    // Store the edge offset and node weight for each node in the chunk and copy the compressed
    // neighborhoods into the actual compressed edge array.
    debug::scoped_time(dbg.copy_time, [&] {
      for (NodeID i = start; i < end; ++i) {
        const EdgeID local_offset = offsets[i - start];

        builder.add_node(i, offset + local_offset);

        if (has_node_weights) [[unlikely]] {
          const NodeID node = node_mapper(i);
          const NodeWeight node_weight = node_weights(node);

          node_weights_array[i] = node_weight;
        }
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
  });

  IF_DBG debug::print_graph_compression_stats(dbg_ets);
  IF_DBG debug::print_compressed_graph_stats(edges_builder_ets);

  return CompressedGraph(builder.build(), std::move(node_weights_array), sorted);
}

} // namespace

[[nodiscard]] CompressedGraph parallel_compress(const CSRGraph &graph);

template <
    typename PermutationMapper,
    typename DegreeMapper,
    typename NodeMapper,
    typename EdgeMapper,
    typename NodeWeightMapper,
    typename EdgeWeightMapper>
[[nodiscard]] CompressedGraph parallel_compress(
    const NodeID num_nodes,
    const EdgeID num_edges,
    const bool has_node_weights,
    const bool has_edge_weights,
    const bool sorted,
    PermutationMapper &&node_mapper,
    DegreeMapper &&degrees,
    NodeMapper &&nodes,
    EdgeMapper &&edges,
    NodeWeightMapper &&node_weights,
    EdgeWeightMapper &&edge_weights
) {
  // To reduce memory usage, we distinguish between graphs with and without edge weights and only
  // store edge weights during compression if they are present.
  if (has_edge_weights) {
    constexpr bool kHasEdgeWeights = true;
    return compute_compressed_graph<kHasEdgeWeights>(
        num_nodes,
        num_edges,
        has_node_weights,
        sorted,
        std::forward<PermutationMapper>(node_mapper),
        std::forward<DegreeMapper>(degrees),
        std::forward<NodeMapper>(nodes),
        std::forward<EdgeMapper>(edges),
        std::forward<NodeWeightMapper>(node_weights),
        std::forward<EdgeWeightMapper>(edge_weights)
    );
  } else {
    constexpr bool kHasEdgeWeights = false;
    return compute_compressed_graph<kHasEdgeWeights>(
        num_nodes,
        num_edges,
        has_node_weights,
        sorted,
        std::forward<PermutationMapper>(node_mapper),
        std::forward<DegreeMapper>(degrees),
        std::forward<NodeMapper>(nodes),
        std::forward<EdgeMapper>(edges),
        std::forward<NodeWeightMapper>(node_weights),
        std::forward<EdgeWeightMapper>(edge_weights)
    );
  }
}

} // namespace kaminpar::shm
