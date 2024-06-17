/*******************************************************************************
 * Sequential and parallel builder for compressed graphs.
 *
 * @file:   compressed_graph_builder.h
 * @author: Daniel Salwasser
 * @date:   03.05.2024
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/compressed_graph.h"

#include "kaminpar-common/datastructures/concurrent_circular_vector.h"
#include "kaminpar-common/datastructures/maxsize_vector.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
SET_DEBUG(false);

class CompressedEdgesBuilder {
  using NodeID = CompressedGraph::NodeID;
  using NodeWeight = CompressedGraph::NodeWeight;
  using EdgeID = CompressedGraph::EdgeID;
  using EdgeWeight = CompressedGraph::EdgeWeight;
  using SignedID = CompressedGraph::SignedID;

public:
  /*!
   * Constructs a new CompressedEdgesBuilder.
   *
   * @param num_nodes The number of nodes of the graph to compress.
   * @param num_edges The number of edges of the graph to compress.
   * @param has_edge_weights Whether the graph to compress has edge weights.
   * @param edge_weights A reference to the edge weights of the compressed graph.
   */
  CompressedEdgesBuilder(
      const NodeID num_nodes,
      const EdgeID num_edges,
      bool has_edge_weights,
      StaticArray<EdgeWeight> &edge_weights
  );

  /*!
   * Constructs a new CompressedEdgesBuilder where the maxmimum degree specifies the number of edges
   * that are compressed at once.
   *
   * @param num_nodes The number of nodes of the graph to compress.
   * @param num_edges The number of edges of the graph to compress.
   * @param max_degree The maximum degree of the graph to compress.
   * @param has_edge_weights Whether the graph to compress has edge weights.
   * @param edge_weights A reference to the edge weights of the compressed graph.
   * @param edge_weights A reference to the edge weights of the compressed graph.
   */
  CompressedEdgesBuilder(
      const NodeID num_nodes,
      const EdgeID num_edges,
      const NodeID max_degree,
      bool has_edge_weights,
      StaticArray<EdgeWeight> &edge_weights
  );

  ~CompressedEdgesBuilder();

  CompressedEdgesBuilder(const CompressedEdgesBuilder &) = delete;
  CompressedEdgesBuilder &operator=(const CompressedEdgesBuilder &) = delete;

  CompressedEdgesBuilder(CompressedEdgesBuilder &&) noexcept = default;

  /*!
   * Initializes/resets the builder.
   *
   * @param first_edge The first edge ID of the first node to be added.
   */
  void init(const EdgeID first_edge);

  /*!
   * Adds the neighborhood of a node. Note that the neighbourhood vector is modified.
   *
   * @param node The node whose neighborhood to add.
   * @param neighbourhood The neighbourhood of the node to add.
   * @return The offset into the compressed edge array of the node.
   */
  template <template <typename> typename Container>
  EdgeID add(const NodeID node, Container<NodeID> &neighbourhood) {
    std::sort(neighbourhood.begin(), neighbourhood.end());
    return add_node(node, neighbourhood);
  }

  /*!
   * Adds the (weighted) neighborhood of a node. Note that the neighbourhood vector is modified.
   *
   * @param node The node whose neighborhood to add.
   * @param neighbourhood The neighbourhood of the node to add.
   * @return The offset into the compressed edge array of the node.
   */
  template <template <typename> typename Container>
  EdgeID add(const NodeID node, Container<std::pair<NodeID, EdgeWeight>> &neighbourhood) {
    std::sort(neighbourhood.begin(), neighbourhood.end(), [](const auto &a, const auto &b) {
      return a.first < b.first;
    });
    return add_node(node, neighbourhood);
  }

  /*!
   * Returns the number of bytes that the compressed data of the added neighborhoods take up.
   *
   * @return The number of bytes that the compressed data of the added neighborhoods take up.
   */
  [[nodiscard]] std::size_t size() const;

  /*!
   * Returns a pointer to the start of the compressed data.
   *
   * @return A pointer to the start of the compressed data.
   */
  [[nodiscard]] const std::uint8_t *compressed_data() const;

  /*!
   * Returns ownership of the compressed data
   *
   * @return Ownership of the compressed data.
   */
  [[nodiscard]] heap_profiler::unique_ptr<std::uint8_t> take_compressed_data();

  [[nodiscard]] std::size_t max_degree() const;
  [[nodiscard]] std::int64_t total_edge_weight() const;

  [[nodiscard]] std::size_t num_high_degree_nodes() const;
  [[nodiscard]] std::size_t num_high_degree_parts() const;
  [[nodiscard]] std::size_t num_interval_nodes() const;
  [[nodiscard]] std::size_t num_intervals() const;

private:
  heap_profiler::unique_ptr<std::uint8_t> _compressed_data_start;
  std::uint8_t *_compressed_data;
  std::size_t _compressed_data_max_size;

  bool _has_edge_weights;
  StaticArray<EdgeWeight> &_edge_weights;

  EdgeID _edge;
  NodeID _max_degree;
  EdgeWeight _total_edge_weight;

  // Graph compression statistics
  std::size_t _num_high_degree_nodes;
  std::size_t _num_high_degree_parts;
  std::size_t _num_interval_nodes;
  std::size_t _num_intervals;

  template <typename Container> EdgeID add_node(const NodeID node, Container &neighbourhood) {
    // The offset into the compressed edge array to the start of the neighbourhood.
    const auto offset = static_cast<EdgeID>(_compressed_data - _compressed_data_start.get());

    const NodeID degree = neighbourhood.size();
    if (degree == 0) {
      return offset;
    }

    _max_degree = std::max(_max_degree, degree);

    // Store a pointer to the first byte of the first edge of this neighborhood. This byte encodes
    // in one of its bits whether interval encoding is used for this node, i.e., whether the nodes
    // has intervals in its neighbourhood.
    std::uint8_t *marked_byte = _compressed_data;

    // Store only the first edge for the source node. The degree can be obtained by determining the
    // difference between the first edge ids of a node and the next node. Additionally, store the
    // first edge as a gap when the isolated nodes are continuously stored at the end of the nodes
    // array.
    const EdgeID first_edge = _edge;
    if constexpr (CompressedGraph::kIntervalEncoding) {
      _compressed_data += marked_varint_encode(first_edge, false, _compressed_data);
    } else {
      _compressed_data += varint_encode(first_edge, _compressed_data);
    }

    // Only increment the edge if edge weights are not stored as otherwise the edge is
    // incremented with each edge weight being added.
    if (!_has_edge_weights) {
      _edge += degree;
    }

    // If high-degree encoding is used then split the neighborhood if the degree crosses a
    // threshold. The neighborhood is split into equally sized parts (except possible the last part)
    // and each part is encoded independently. Furthermore, the offset at which the part is encoded
    // is also stored.
    if constexpr (CompressedGraph::kHighDegreeEncoding) {
      const bool split_neighbourhood = degree >= CompressedGraph::kHighDegreeThreshold;

      if (split_neighbourhood) {
        const NodeID part_count = math::div_ceil(degree, CompressedGraph::kHighDegreePartLength);
        const NodeID last_part_length = ((degree % CompressedGraph::kHighDegreePartLength) == 0)
                                            ? CompressedGraph::kHighDegreePartLength
                                            : (degree % CompressedGraph::kHighDegreePartLength);

        uint8_t *part_ptr = _compressed_data;
        _compressed_data += sizeof(NodeID) * part_count;

        for (NodeID i = 0; i < part_count; ++i) {
          const bool last_part = (i + 1) == part_count;
          const NodeID part_length =
              last_part ? last_part_length : CompressedGraph::kHighDegreePartLength;

          auto part_begin = neighbourhood.begin() + i * CompressedGraph::kHighDegreePartLength;
          auto part_end = part_begin + part_length;

          std::uint8_t *cur_part_ptr = part_ptr + sizeof(NodeID) * i;
          *((NodeID *)cur_part_ptr) = static_cast<NodeID>(_compressed_data - part_ptr);

          using Neighbour = typename Container::value_type;
          add_edges(node, nullptr, std::span<Neighbour>(part_begin, part_end));
        }

        _num_high_degree_nodes += 1;
        _num_high_degree_parts += part_count;
        return offset;
      }
    }

    add_edges(node, marked_byte, std::forward<decltype(neighbourhood)>(neighbourhood));
    return offset;
  }

  template <typename Container>
  void add_edges(const NodeID node, std::uint8_t *marked_byte, Container &&neighbourhood) {
    using Neighbour = std::remove_reference_t<Container>::value_type;
    constexpr bool kHasEdgeWeights = std::is_same_v<Neighbour, std::pair<NodeID, EdgeWeight>>;

    const auto store_edge_weight = [&](const EdgeWeight edge_weight) {
      _edge_weights[_edge++] = edge_weight;
      _total_edge_weight += edge_weight;
    };

    const auto fetch_adjacent_node = [&](const NodeID i) {
      if constexpr (kHasEdgeWeights) {
        return neighbourhood[i].first;
      } else {
        return neighbourhood[i];
      }
    };

    const auto set_adjacent_node = [&](const NodeID i, const NodeID value) {
      if constexpr (kHasEdgeWeights) {
        neighbourhood[i].first = value;
      } else {
        neighbourhood[i] = value;
      }
    };

    NodeID local_degree = neighbourhood.size();

    // Find intervals [i, j] of consecutive adjacent nodes i, i + 1, ..., j - 1, j of length at
    // least kIntervalLengthTreshold. Instead of storing all nodes, only encode the left extreme i
    // and the length j - i + 1. Left extremes are stored using the differences between each left
    // extreme and the previous right extreme minus 2 (because there must be at least one integer
    // between the end of an interval and the beginning of the next one), except the first left
    // extreme, which is stored directly. The lengths are decremented by kIntervalLengthTreshold,
    // the minimum length of an interval.
    if constexpr (CompressedGraph::kIntervalEncoding) {
      NodeID interval_count = 0;

      // Save the pointer to the interval count and skip the amount of bytes needed to store the
      // interval count as we can only determine the amount of intervals after finding all of
      // them.
      std::uint8_t *interval_count_ptr = _compressed_data;
      _compressed_data += sizeof(NodeID);

      if (local_degree >= CompressedGraph::kIntervalLengthTreshold) {
        NodeID interval_len = 1;
        NodeID previous_right_extreme = 2;
        NodeID prev_adjacent_node = fetch_adjacent_node(0);

        for (NodeID i = 1; i < neighbourhood.size(); ++i) {
          const NodeID adjacent_node = fetch_adjacent_node(i);

          if (prev_adjacent_node + 1 == adjacent_node) {
            ++interval_len;

            // The interval ends if there are no more nodes or the next node is not the increment of
            // the current node.
            if (i + 1 == neighbourhood.size() || fetch_adjacent_node(i + 1) != adjacent_node + 1) {
              if (interval_len >= CompressedGraph::kIntervalLengthTreshold) {
                const NodeID left_extreme = adjacent_node + 1 - interval_len;
                const NodeID left_extreme_gap = left_extreme + 2 - previous_right_extreme;
                const NodeID interval_length_gap =
                    interval_len - CompressedGraph::kIntervalLengthTreshold;

                _compressed_data += varint_encode(left_extreme_gap, _compressed_data);
                _compressed_data += varint_encode(interval_length_gap, _compressed_data);

                for (NodeID j = 0; j < interval_len; ++j) {
                  const NodeID k = i + 1 + j - interval_len;

                  // Set the adjacent node to a special value, which indicates for the gap encoder
                  // that the node has been encoded through an interval.
                  set_adjacent_node(k, std::numeric_limits<NodeID>::max());

                  if constexpr (kHasEdgeWeights) {
                    if (_has_edge_weights) {
                      const EdgeWeight edge_weight = neighbourhood[k].second;
                      store_edge_weight(edge_weight);
                    }
                  }
                }

                previous_right_extreme = adjacent_node;

                local_degree -= interval_len;
                interval_count += 1;
              }

              interval_len = 1;
            }
          }

          prev_adjacent_node = adjacent_node;
        }
      }

      // If intervals have been encoded store the interval count and set the bit in the marked byte
      // indicating that interval encoding has been used for the neighbourhood if the marked byte is
      // given. Otherwise, fix the amount of bytes stored as we don't store the interval count if no
      // intervals have been encoded.
      if (marked_byte == nullptr) {
        *((NodeID *)interval_count_ptr) = interval_count;
      } else if (interval_count > 0) {
        *((NodeID *)interval_count_ptr) = interval_count;
        *marked_byte |= 0b01000000;
      } else {
        _compressed_data -= sizeof(NodeID);
      }

      if (interval_count > 0) {
        _num_interval_nodes += 1;
        _num_intervals += interval_count;
      }

      // If all incident edges have been compressed using intervals then gap encoding cannot be
      // applied.
      if (local_degree == 0) {
        return;
      }
    }

    // Store the remaining adjacent nodes using gap encoding. That is instead of directly storing
    // the nodes v_1, v_2, ..., v_{k - 1}, v_k, store the gaps v_1 - u, v_2 - v_1 - 1, ..., v_k -
    // v_{k - 1} - 1 between the nodes, where u is the source node. Note that all gaps except the
    // first one have to be positive as we sorted the nodes in ascending order. Thus, only for the
    // first gap the sign is additionally stored.
    NodeID i = 0;

    // Go to the first adjacent node that has not been encoded through an interval.
    if constexpr (CompressedGraph::kIntervalEncoding) {
      while (fetch_adjacent_node(i) == std::numeric_limits<NodeID>::max()) {
        i += 1;
      }
    }

    const NodeID first_adjacent_node = fetch_adjacent_node(i);
    const SignedID first_gap = first_adjacent_node - static_cast<SignedID>(node);
    _compressed_data += signed_varint_encode(first_gap, _compressed_data);

    if constexpr (kHasEdgeWeights) {
      if (_has_edge_weights) {
        const EdgeWeight first_edge_weight = neighbourhood[i].second;
        store_edge_weight(first_edge_weight);
      }
    }

    i += 1;

    VarIntRunLengthEncoder<NodeID> rl_encoder(_compressed_data);
    VarIntStreamEncoder<NodeID> sv_encoder(_compressed_data, local_degree - 1);

    NodeID prev_adjacent_node = first_adjacent_node;
    while (i < neighbourhood.size()) {
      const NodeID adjacent_node = fetch_adjacent_node(i);

      // Skip the adjacent node since it has been encoded through an interval.
      if constexpr (CompressedGraph::kIntervalEncoding) {
        if (adjacent_node == std::numeric_limits<NodeID>::max()) {
          i += 1;
          continue;
        }
      }

      const NodeID gap = adjacent_node - prev_adjacent_node - 1;
      if constexpr (CompressedGraph::kRunLengthEncoding) {
        _compressed_data += rl_encoder.add(gap);
      } else if constexpr (CompressedGraph::kStreamEncoding) {
        _compressed_data += sv_encoder.add(gap);
      } else {
        _compressed_data += varint_encode(gap, _compressed_data);
      }

      if constexpr (kHasEdgeWeights) {
        if (_has_edge_weights) {
          const EdgeWeight edge_weight = neighbourhood[i].second;
          store_edge_weight(edge_weight);
        }
      }

      prev_adjacent_node = adjacent_node;
      i += 1;
    }

    if constexpr (CompressedGraph::kRunLengthEncoding) {
      rl_encoder.flush();
    } else if constexpr (CompressedGraph::kStreamEncoding) {
      sv_encoder.flush();
    }
  }
};

/*!
 * A sequential builder that constructs compressed graphs.
 */
class CompressedGraphBuilder {
  using NodeID = CompressedGraph::NodeID;
  using NodeWeight = CompressedGraph::NodeWeight;
  using EdgeID = CompressedGraph::EdgeID;
  using EdgeWeight = CompressedGraph::EdgeWeight;
  using SignedID = CompressedGraph::SignedID;

public:
  /*!
   * Compresses a graph in compressed sparse row format.
   *
   * @param graph The graph to compress.
   * @return The compressed input graph.
   */
  static CompressedGraph compress(const CSRGraph &graph);

  /*!
   * Constructs a new CompressedGraphBuilder.
   *
   * @param node_count The number of nodes of the graph to compress.
   * @param edge_count The number of edges of the graph to compress.
   * @param has_node_weights Whether node weights are stored.
   * @param has_edge_weights Whether edge weights are stored.
   * @param sorted Whether the nodes to add are stored in degree-bucket order.
   */
  CompressedGraphBuilder(
      const NodeID node_count,
      const EdgeID edge_count,
      const bool has_node_weights,
      const bool has_edge_weights,
      const bool sorted
  );

  /*!
   * Adds a node to the compressed graph. Note that the neighbourhood vector is modified.
   *
   * @param node The node to add.
   * @param neighbourhood The neighbourhood of the node to add.
   */
  void add_node(const NodeID node, std::vector<std::pair<NodeID, EdgeWeight>> &neighbourhood);

  /*!
   * Adds a node weight to the compressed graph.
   *
   * @param node The node whose weight to add.
   * @param weight The weight to store.
   */
  void add_node_weight(const NodeID node, const NodeWeight weight);

  /*!
   * Builds the compressed graph. The builder must then be reinitialized in order to compress
   * another graph.
   *
   * @return The compressed graph that has been build.
   */
  CompressedGraph build();

  /*!
   * Returns the used memory of the compressed edge array.
   *
   * @return The used memory of the compressed edge array.
   */
  [[nodiscard]] std::size_t currently_used_memory() const;

  /*!
   * Returns the total weight of the nodes that have been added.
   *
   * @return The total weight of the nodes that have been added.
   */
  [[nodiscard]] std::int64_t total_node_weight() const;

  /*!
   * Returns the total weight of the edges that have been added.
   *
   * @return The total weight of the edges that have been added.
   */
  [[nodiscard]] std::int64_t total_edge_weight() const;

private:
  // The arrays that store information about the compressed graph
  CompactStaticArray<EdgeID> _nodes;
  bool _sorted; // Whether the nodes of the graph are stored in degree-bucket order

  CompressedEdgesBuilder _compressed_edges_builder;
  EdgeID _num_edges;

  StaticArray<NodeWeight> _node_weights;
  StaticArray<EdgeWeight> _edge_weights;

  // Statistics about the graph
  bool _store_node_weights;
  std::int64_t _total_node_weight;
};

class ParallelCompressedGraphBuilder {
  using NodeID = CompressedGraph::NodeID;
  using NodeWeight = CompressedGraph::NodeWeight;
  using EdgeID = CompressedGraph::EdgeID;
  using EdgeWeight = CompressedGraph::EdgeWeight;

public:
  /*!
   * Compresses a graph.
   *
   * @param num_nodes The number of nodes of the graph to compress.
   * @param num_edges The number of edges of the graph to compress.
   * @param has_node_weights Whether node weights are stored.
   * @param has_edge_weights Whether edge weights are stored.
   * @param sorted Whether the nodes are stored in degree-bucket order.
   * @param node_mapper Function that maps old node IDs to (possibly) new ones.
   * @param degrees Function that returns the degree of a (remapped) node.
   * @param nodes Function that returns the first edge of a node.
   * @param edges Function that returns the (remapped) adjacent node of an edge.
   * @param node_weights Function that returns the weight of a node.
   * @param edge_weights Function that returns the weight of an edge.
   * @return The compressed graph.
   */
  template <
      typename PermutationMapper,
      typename DegreeMapper,
      typename NodeMapper,
      typename EdgeMapper,
      typename NodeWeightMapper,
      typename EdgeWeightMapper>
  [[nodiscard]] static CompressedGraph compress(
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
  );

  /*!
   * Compresses a graph stored in compressed sparse row format.
   *
   * @param graph The graph to compress.
   * @return The compressed graph.
   */
  [[nodiscard]] static CompressedGraph compress(const CSRGraph &graph);

  /*!
   * Initializes the builder by allocating memory for the various arrays.
   *
   * @param num_nodes The number of nodes of the graph to compress.
   * @param num_edges The number of edges of the graph to compress.
   * @param has_node_weights Whether node weights are stored.
   * @param has_edge_weights Whether edge weights are stored.
   * @param sorted Whether the nodes to add are stored in degree-bucket order.
   */
  ParallelCompressedGraphBuilder(
      const NodeID num_nodes,
      const EdgeID num_edges,
      const bool has_node_weights,
      const bool has_edge_weights,
      const bool sorted
  );

  /*!
   * Adds a node to the compressed graph.
   *
   * @param node The node to add.
   * @param offset The offset into the compressed edge array at which the compressed neighborhood
   * of the node is stored.
   */
  void add_node(const NodeID node, const EdgeID offset);

  /**
   * Adds compressed neighborhoods of possible multiple consecutive nodes to the compressed graph.
   *
   * @param offset The offset into the compressed edge array at which the compressed neighborhoods
   * are stored.
   * @param length The length in bytes of the compressed neighborhoods to store.
   * @param data A pointer to the start of the compressed neighborhoods to copy.
   */
  void add_compressed_edges(const EdgeID offset, const EdgeID length, const std::uint8_t *data);

  /*!
   * Adds a node weight to the compressed graph.
   *
   * @param node The node whose weight to add.
   * @param weight The weight to store.
   */
  void add_node_weight(const NodeID node, const NodeWeight weight);

  /*!
   * Returns a reference to the edge weights of the compressed graph.
   *
   * @return A reference to the edge weights of the compressed graph.
   */
  [[nodiscard]] StaticArray<EdgeWeight> &edge_weights();

  /*!
   * Adds (cummulative) statistics about nodes of the compressed graph.
   */
  void record_local_statistics(
      NodeID max_degree,
      NodeWeight node_weight,
      EdgeWeight edge_weight,
      std::size_t num_high_degree_nodes,
      std::size_t num_high_degree_parts,
      std::size_t num_interval_nodes,
      std::size_t num_intervals
  );

  /*!
   * Finalizes the compressed graph. Note that all nodes, compressed neighborhoods, node weights
   * and edge weights have to be added at this point.
   *
   * @return The resulting compressed graph.
   */
  [[nodiscard]] CompressedGraph build();

private:
  // The arrays that store information about the compressed graph
  CompactStaticArray<EdgeID> _nodes;
  bool _sorted; // Whether the nodes of the graph are stored in degree-bucket order

  heap_profiler::unique_ptr<std::uint8_t> _compressed_edges;
  EdgeID _compressed_edges_size;
  EdgeID _num_edges;

  StaticArray<NodeWeight> _node_weights;
  StaticArray<EdgeWeight> _edge_weights;

  NodeID _max_degree;
  NodeWeight _total_node_weight;
  EdgeWeight _total_edge_weight;

  // Statistics about graph compression
  std::size_t _num_high_degree_nodes;
  std::size_t _num_high_degree_parts;
  std::size_t _num_interval_nodes;
  std::size_t _num_intervals;
};

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

void print_stats(const auto &stats_ets) {
  DBG << "Chunk distribution:";

  std::size_t cur_thread = 0;
  for (const auto &stats : stats_ets) {
    DBG << "t" << ++cur_thread << ": " << stats.num_chunks;
  }

  DBG << "Edge distribution:";

  cur_thread = 0;
  for (const auto &stats : stats_ets) {
    DBG << "t" << ++cur_thread << ": " << stats.num_edges;
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

    DBG << "t" << ++cur_thread << ": " << to_sec(stats.compression_time) << ' '
        << to_sec(stats.sync_time) << ' ' << to_sec(stats.copy_time);
  }

  DBG << "sum: " << to_sec(total_time_compression) << ' ' << to_sec(total_time_sync) << ' '
      << to_sec(total_time_copy);
}

} // namespace debug

namespace {

template <
    bool kHasEdgeWeights,
    typename PermutationMapper,
    typename DegreeMapper,
    typename NodeMapper,
    typename EdgeMapper,
    typename NodeWeightMapper,
    typename EdgeWeightMapper>
CompressedGraph compute_compressed_graph(
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
  constexpr std::size_t kNumChunks = 5000;
  const EdgeID max_chunk_size = num_edges / kNumChunks;
  std::vector<std::tuple<NodeID, NodeID, EdgeID>> chunks;

  NodeID max_degree = 0;
  TIMED_SCOPE("Compute chunks") {
    NodeID cur_chunk_start = 0;
    EdgeID cur_chunk_size = 0;
    EdgeID cur_first_edge = 0;
    for (NodeID i = 0; i < num_nodes; ++i) {
      NodeID node = node_mapper(i);

      const NodeID degree = degrees(node);
      max_degree = std::max(max_degree, degree);

      cur_chunk_size += degree;
      if (cur_chunk_size >= max_chunk_size) {
        if (cur_chunk_start == i) {
          chunks.emplace_back(cur_chunk_start, i + 1, cur_first_edge);

          cur_chunk_start = i + 1;
          cur_first_edge += degree;
          cur_chunk_size = 0;
        } else {
          chunks.emplace_back(cur_chunk_start, i, cur_first_edge);

          cur_chunk_start = i;
          cur_first_edge += cur_chunk_size - degree;
          cur_chunk_size = degree;
        }
      }
    }

    if (cur_chunk_start != num_nodes) {
      chunks.emplace_back(cur_chunk_start, num_nodes, cur_first_edge);
    }
  };

  // Initializes the data structures used to build the compressed graph in parallel.
  ParallelCompressedGraphBuilder builder(
      num_nodes, num_edges, has_node_weights, kHasEdgeWeights, sorted
  );

  const std::size_t max_capacity = std::max<std::size_t>(max_chunk_size, max_degree);

  tbb::enumerable_thread_specific<MaxSizeVector<EdgeID>> offsets_ets([&] {
    return MaxSizeVector<EdgeID>(max_capacity);
  });

  using Neighbourhood = std::conditional_t<
      kHasEdgeWeights,
      MaxSizeVector<std::pair<NodeID, EdgeWeight>>,
      MaxSizeVector<NodeID>>;
  tbb::enumerable_thread_specific<Neighbourhood> neighbourhood_ets([&] {
    return Neighbourhood(max_capacity);
  });

  tbb::enumerable_thread_specific<CompressedEdgesBuilder> neighbourhood_builder_ets([&] {
    return CompressedEdgesBuilder(
        num_nodes, num_edges, max_degree, kHasEdgeWeights, builder.edge_weights()
    );
  });

  const std::size_t num_threads = tbb::this_task_arena::max_concurrency();
  ConcurrentCircularVectorMutex<NodeID, EdgeID> buffer(num_threads);

  tbb::enumerable_thread_specific<debug::Stats> dbg_ets;
  tbb::parallel_for<NodeID>(0, chunks.size(), [&](const auto) {
    auto &dbg = dbg_ets.local();
    IF_DBG dbg.num_chunks++;

    auto &offsets = offsets_ets.local();
    auto &neighbourhood = neighbourhood_ets.local();
    CompressedEdgesBuilder &neighbourhood_builder = neighbourhood_builder_ets.local();

    const NodeID chunk = buffer.next();
    const auto [start, end, first_edge] = chunks[chunk];

    NodeWeight local_node_weight = 0;
    neighbourhood_builder.init(first_edge);

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

        const EdgeID local_offset = neighbourhood_builder.add(i, neighbourhood);
        offsets.push_back(local_offset);

        neighbourhood.clear();
      }
    });

    // Wait for the parallel tasks that process the previous chunks to finish.
    const EdgeID offset = debug::scoped_time(dbg.sync_time, [&] {
      const EdgeID compressed_neighborhoods_size = neighbourhood_builder.size();
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
          local_node_weight += node_weight;

          builder.add_node_weight(i, node_weight);
        }
      }
      offsets.clear();

      builder.add_compressed_edges(
          offset, neighbourhood_builder.size(), neighbourhood_builder.compressed_data()
      );

      builder.record_local_statistics(
          neighbourhood_builder.max_degree(),
          local_node_weight,
          neighbourhood_builder.total_edge_weight(),
          neighbourhood_builder.num_high_degree_nodes(),
          neighbourhood_builder.num_high_degree_parts(),
          neighbourhood_builder.num_interval_nodes(),
          neighbourhood_builder.num_intervals()
      );
    });
  });

  IF_DBG debug::print_stats(dbg_ets);

  return builder.build();
}

} // namespace

template <
    typename PermutationMapper,
    typename DegreeMapper,
    typename NodeMapper,
    typename EdgeMapper,
    typename NodeWeightMapper,
    typename EdgeWeightMapper>
CompressedGraph ParallelCompressedGraphBuilder::compress(
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
