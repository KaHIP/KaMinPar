/*******************************************************************************
 * Compressed neighborhoods builder.
 *
 * @file:   compressed_neighborhoods_builder.h
 * @author: Daniel Salwasser
 * @date:   09.07.2024
 ******************************************************************************/
#pragma once

#include "kaminpar-common/datastructures/compact_static_array.h"
#include "kaminpar-common/graph_compression/compressed_edges_builder.h"
#include "kaminpar-common/graph_compression/compressed_neighborhoods.h"

namespace kaminpar {

template <typename NodeID, typename EdgeID, typename EdgeWeight>
class CompressedNeighborhoodsBuilder {
  using CompressedEdgesBuilder = kaminpar::CompressedEdgesBuilder<NodeID, EdgeID, EdgeWeight>;
  using CompressedNeighborhoods = kaminpar::CompressedNeighborhoods<NodeID, EdgeID, EdgeWeight>;

public:
  /*!
   * Constructs a new CompressedNeighborhoodsBuilder.
   *
   * @param num_nodes The number of nodes of the graph to compress.
   * @param num_edges The number of edges of the graph to compress.
   * @param has_edge_weights Whether edge weights are stored.
   */
  CompressedNeighborhoodsBuilder(
      const NodeID num_nodes, const EdgeID num_edges, const bool has_edge_weights
  )
      : _compressed_edges_builder(
            CompressedEdgesBuilder::num_edges_tag,
            num_nodes,
            num_edges,
            has_edge_weights,
            _edge_weights
        ),
        _num_edges(num_edges),
        _has_edge_weights(has_edge_weights) {

    const std::size_t max_size = CompressedEdgesBuilder::compressed_edge_array_max_size(
        num_nodes, num_edges, has_edge_weights
    );
    _nodes.resize(math::byte_width(max_size), num_nodes + 1);
    _compressed_edges_builder.init(0);

    if constexpr (!CompressedNeighborhoods::kCompressEdgeWeights) {
      if (has_edge_weights) {
        _edge_weights.resize(num_edges, static_array::noinit);
      }
    }
  }

  /**
   * Sets the number of edges of the graph to compress.
   *
   * @param num_edges The number of edges of the graph to compress.
   */
  void set_num_edges(const EdgeID num_edges) {
    _num_edges = num_edges;
  }

  /*!
   * Adds the (possibly weighted) neighborhood of a node. Note that the neighbourhood vector is
   * modified.
   *
   * @param node The node whose neighborhood to add.
   * @param neighbourhood The neighbourhood of the node to add.
   */
  template <typename Container> void add(const NodeID node, Container &neighbourhood) {
    KASSERT(node + 1 < _nodes.size());

    const EdgeID offset = _compressed_edges_builder.add(node, neighbourhood);
    _nodes.write(node, offset);
  }

  /*!
   * Builds the compressed neighborhoods. The builder must then be reinitialized in order to
   * compress further neighborhoods.
   *
   * @return The compressed neighborhoods that have been build.
   */
  CompressedNeighborhoods build() {
    std::size_t compressed_edges_size = _compressed_edges_builder.size();
    auto compressed_edges = _compressed_edges_builder.take_compressed_data();

    // Store in the last entry of the node array the offset one after the last byte belonging to the
    // last node.
    _nodes.write(_nodes.size() - 1, static_cast<EdgeID>(compressed_edges_size));

    // Store at the end of the compressed edge array the (gap of the) id of the last edge. This
    // ensures that the the degree of the last node can be computed from the difference between the
    // last two first edge ids.
    std::uint8_t *_compressed_edges_end = compressed_edges.get() + compressed_edges_size;
    const EdgeID last_edge = _num_edges;
    if constexpr (CompressedNeighborhoods::kIntervalEncoding) {
      compressed_edges_size += marked_varint_encode(last_edge, false, _compressed_edges_end);
    } else {
      compressed_edges_size += varint_encode(last_edge, _compressed_edges_end);
    }

    // Add an additional 15 bytes to the compressed edge array when stream encoding is enabled to
    // avoid a possible segmentation fault as the stream decoder reads 16-byte chunks.
    if constexpr (CompressedNeighborhoods::kStreamVByteEncoding) {
      compressed_edges_size += 15;
    }

    if constexpr (kHeapProfiling) {
      heap_profiler::HeapProfiler::global().record_alloc(
          compressed_edges.get(), compressed_edges_size
      );
    }

    return CompressedNeighborhoods(
        std::move(_nodes),
        StaticArray<std::uint8_t>(compressed_edges_size, std::move(compressed_edges)),
        std::move(_edge_weights),
        _compressed_edges_builder.max_degree(),
        _num_edges,
        _has_edge_weights,
        _has_edge_weights ? _compressed_edges_builder.total_edge_weight() : _num_edges,
        _compressed_edges_builder.num_high_degree_nodes(),
        _compressed_edges_builder.num_high_degree_parts(),
        _compressed_edges_builder.num_interval_nodes(),
        _compressed_edges_builder.num_intervals()
    );
  }

  /*!
   * Returns the used memory of the compressed neighborhoods.
   *
   * @return The used memory of the compressed neighborhoods.
   */
  [[nodiscard]] std::size_t currently_used_memory() const {
    return _nodes.memory_space() + _compressed_edges_builder.size();
  }

  /*!
   * Returns the total edge weight.
   *
   * @return The total edge weight.
   */
  [[nodiscard]] std::int64_t total_edge_weight() const {
    return _compressed_edges_builder.total_edge_weight();
  }

private:
  CompactStaticArray<EdgeID> _nodes;
  CompressedEdgesBuilder _compressed_edges_builder;
  StaticArray<EdgeWeight> _edge_weights;
  EdgeID _num_edges;
  bool _has_edge_weights;
};

template <typename NodeID, typename EdgeID, typename EdgeWeight>
class ParallelCompressedNeighborhoodsBuilder {
  using CompressedEdgesBuilder = kaminpar::CompressedEdgesBuilder<NodeID, EdgeID, EdgeWeight>;
  using CompressedNeighborhoods = kaminpar::CompressedNeighborhoods<NodeID, EdgeID, EdgeWeight>;

public:
  /*!
   * Constructs a new ParallelCompressedNeighborhoodsBuilder.
   *
   * @param num_nodes The number of nodes of the graph to compress.
   * @param num_edges The number of edges of the graph to compress.
   * @param has_edge_weights Whether edge weights are stored.
   */
  ParallelCompressedNeighborhoodsBuilder(
      const NodeID num_nodes, const EdgeID num_edges, const bool has_edge_weights
  )
      : _num_edges(num_edges),
        _max_degree(0),
        _has_edge_weights(has_edge_weights),
        _total_edge_weight(0),
        _num_high_degree_nodes(0),
        _num_high_degree_parts(0),
        _num_interval_nodes(0),
        _num_intervals(0) {
    const std::size_t max_size = CompressedEdgesBuilder::compressed_edge_array_max_size(
        num_nodes, num_edges, has_edge_weights
    );
    _nodes.resize(math::byte_width(max_size), num_nodes + 1);
    _compressed_edges = heap_profiler::overcommit_memory<std::uint8_t>(max_size);
    _compressed_edges_size = 0;

    if constexpr (!CompressedNeighborhoods::kCompressEdgeWeights) {
      if (has_edge_weights) {
        _edge_weights.resize(num_edges, static_array::noinit);
      }
    }
  }

  /*!
   * Adds a node to the compressed neighborhoods.
   *
   * @param node The node to add.
   * @param offset The offset into the compressed edge array at which the compressed neighborhood
   * of the node is stored.
   */
  void add_node(const NodeID node, const EdgeID offset) {
    _nodes.write(node, offset);
  }

  /**
   * Adds compressed neighborhoods of possible multiple consecutive nodes to the compressed graph.
   *
   * @param offset The offset into the compressed edge array at which the compressed neighborhoods
   * are stored.
   * @param length The length in bytes of the compressed neighborhoods to store.
   * @param data A pointer to the start of the compressed neighborhoods to copy.
   */
  void add_compressed_edges(const EdgeID offset, const EdgeID length, const std::uint8_t *data) {
    __atomic_fetch_add(&_compressed_edges_size, length, __ATOMIC_RELAXED);
    std::memcpy(_compressed_edges.get() + offset, data, length);
  }

  /*!
   * Adds (cummulative) statistics about nodes of the compressed graph.
   */
  void record_local_statistics(
      NodeID max_degree,
      EdgeWeight edge_weight,
      std::size_t num_high_degree_nodes,
      std::size_t num_high_degree_parts,
      std::size_t num_interval_nodes,
      std::size_t num_intervals
  ) {
    NodeID global_max_degree = __atomic_load_n(&_max_degree, __ATOMIC_RELAXED);
    while (max_degree > global_max_degree) {
      const bool success = __atomic_compare_exchange_n(
          &_max_degree, &global_max_degree, max_degree, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED
      );

      if (success) {
        break;
      }
    }

    __atomic_fetch_add(&_total_edge_weight, edge_weight, __ATOMIC_RELAXED);

    __atomic_fetch_add(&_num_high_degree_nodes, num_high_degree_nodes, __ATOMIC_RELAXED);
    __atomic_fetch_add(&_num_high_degree_parts, num_high_degree_parts, __ATOMIC_RELAXED);
    __atomic_fetch_add(&_num_interval_nodes, num_interval_nodes, __ATOMIC_RELAXED);
    __atomic_fetch_add(&_num_intervals, num_intervals, __ATOMIC_RELAXED);
  }

  /*!
   * Finalizes the compressed neighborhoods. Note that all nodes and compressed neighborhoods have
   * to be added at this point. The builder must then be reinitialized in order to compress further
   * neighborhoods.
   *
   * @return The compressed neighborhoods that have been build.
   */
  [[nodiscard]] CompressedNeighborhoods build() {
    // Store in the last entry of the node array the offset one after the last byte belonging to the
    // last node.
    _nodes.write(_nodes.size() - 1, _compressed_edges_size);

    // Store at the end of the compressed edge array the (gap of the) id of the last edge. This
    // ensures that the the degree of the last node can be computed from the difference between the
    // last two first edge ids.
    std::uint8_t *_compressed_edges_end = _compressed_edges.get() + _compressed_edges_size;
    const EdgeID last_edge = _num_edges;
    if constexpr (CompressedNeighborhoods::kIntervalEncoding) {
      _compressed_edges_size += marked_varint_encode(last_edge, false, _compressed_edges_end);
    } else {
      _compressed_edges_size += varint_encode(last_edge, _compressed_edges_end);
    }

    // Add an additional 15 bytes to the compressed edge array when stream encoding is enabled to
    // avoid a possible segmentation fault as the stream decoder reads 16-byte chunks.
    if constexpr (CompressedNeighborhoods::kStreamVByteEncoding) {
      _compressed_edges_size += 15;
    }

    if constexpr (kHeapProfiling) {
      heap_profiler::HeapProfiler::global().record_alloc(
          _compressed_edges.get(), _compressed_edges_size
      );
    }

    return CompressedNeighborhoods(
        std::move(_nodes),
        StaticArray<std::uint8_t>(_compressed_edges_size, std::move(_compressed_edges)),
        std::move(_edge_weights),
        _max_degree,
        _num_edges,
        _has_edge_weights,
        _has_edge_weights ? _total_edge_weight : _num_edges,
        _num_high_degree_nodes,
        _num_high_degree_parts,
        _num_interval_nodes,
        _num_intervals
    );
  }

  /*!
   * Returns a reference to the edge weights.
   *
   * Note that it is only valid when edge weight compression is disabled and when the graph has edge
   * weights.
   *
   * @return A reference to the edge weights.
   */
  [[nodiscard]] StaticArray<EdgeWeight> &edge_weights() {
    return _edge_weights;
  }

private:
  CompactStaticArray<EdgeID> _nodes;
  heap_profiler::unique_ptr<std::uint8_t> _compressed_edges;
  EdgeID _compressed_edges_size;

  EdgeID _num_edges;
  NodeID _max_degree;

  bool _has_edge_weights;
  EdgeWeight _total_edge_weight;
  StaticArray<EdgeWeight> _edge_weights;

  // Statistics about graph compression
  std::size_t _num_high_degree_nodes;
  std::size_t _num_high_degree_parts;
  std::size_t _num_interval_nodes;
  std::size_t _num_intervals;
};

} // namespace kaminpar
