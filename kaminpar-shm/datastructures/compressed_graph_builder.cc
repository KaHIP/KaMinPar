/*******************************************************************************
 * Sequential and parallel builder for compressed graphs.
 *
 * @file:   compressed_graph_builder.cc
 * @author: Daniel Salwasser
 * @date:   03.05.2024
 ******************************************************************************/
#include "kaminpar-shm/datastructures/compressed_graph_builder.h"

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <span>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/concurrent_circular_vector.h"
#include "kaminpar-common/heap_profiler.h"

namespace kaminpar::shm {

namespace {

template <bool kActualNumEdges = true>
[[nodiscard]] std::size_t
compressed_edge_array_max_size(const NodeID num_nodes, const EdgeID num_edges) {
  std::size_t edge_id_width;
  if constexpr (kActualNumEdges) {
    if constexpr (CompressedGraph::kIntervalEncoding) {
      edge_id_width = marked_varint_length(num_edges);
    } else {
      edge_id_width = varint_length(num_edges);
    }
  } else {
    edge_id_width = varint_max_length<EdgeID>();
  }

  std::size_t max_size = num_nodes * edge_id_width + num_edges * varint_length(num_nodes);

  if constexpr (CompressedGraph::kHighDegreeEncoding) {
    if constexpr (CompressedGraph::kIntervalEncoding) {
      max_size += 2 * num_nodes * varint_max_length<NodeID>();
    } else {
      max_size += num_nodes * varint_max_length<NodeID>();
    }

    max_size += (num_edges / CompressedGraph::kHighDegreePartLength) * varint_max_length<NodeID>();
  }

  return max_size;
}

} // namespace

CompressedEdgesBuilder::CompressedEdgesBuilder(
    const NodeID num_nodes,
    const EdgeID num_edges,
    bool has_edge_weights,
    StaticArray<EdgeWeight> &edge_weights
)
    : _has_edge_weights(has_edge_weights),
      _edge_weights(edge_weights) {
  const std::size_t max_size = compressed_edge_array_max_size(num_nodes, num_edges);
  _compressed_data_start = heap_profiler::overcommit_memory<std::uint8_t>(max_size);
}

CompressedEdgesBuilder::CompressedEdgesBuilder(
    const NodeID num_nodes,
    const EdgeID num_edges,
    const NodeID max_degree,
    bool has_edge_weights,
    StaticArray<EdgeWeight> &edge_weights
)
    : _has_edge_weights(has_edge_weights),
      _edge_weights(edge_weights) {
  const std::size_t max_size = compressed_edge_array_max_size<false>(num_nodes, max_degree);
  _compressed_data_start = heap_profiler::overcommit_memory<std::uint8_t>(max_size);
}

void CompressedEdgesBuilder::init(const EdgeID first_edge) {
  _compressed_data = _compressed_data_start.get();

  _edge = first_edge;
  _max_degree = 0;
  _total_edge_weight = 0;

  _num_high_degree_nodes = 0;
  _num_high_degree_parts = 0;
  _num_interval_nodes = 0;
  _num_intervals = 0;
}

EdgeID CompressedEdgesBuilder::add(
    const NodeID node, std::vector<std::pair<NodeID, EdgeWeight>> &neighbourhood
) {
  // The offset into the compressed edge array of the start of the neighbourhood.
  const auto offset = static_cast<EdgeID>(_compressed_data - _compressed_data_start.get());

  const NodeID degree = neighbourhood.size();
  if (degree == 0) {
    return offset;
  }

  _max_degree = std::max(_max_degree, degree);

  // Store a pointer to the first byte of the first edge of this neighborhood. This byte encodes in
  // one of its bits whether interval encoding is used for this node, i.e., whether the nodes has
  // intervals in its neighbourhood.
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

  // Sort the adjacent nodes in ascending order.
  std::sort(neighbourhood.begin(), neighbourhood.end(), [](const auto &a, const auto &b) {
    return a.first < b.first;
  });

  // If high-degree encoding is used then split the neighborhood if the degree crosses a threshold.
  // The neighborhood is split into equally sized parts (except possible the last part) and each
  // part is encoded independently. Furthermore, the offset at which the part is encoded is also
  // stored.
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

        std::span<std::pair<NodeID, EdgeWeight>> part_neighbourhood(part_begin, part_end);
        add_edges(node, nullptr, part_neighbourhood);
      }

      _num_high_degree_nodes += 1;
      _num_high_degree_parts += part_count;
      return offset;
    }
  }

  add_edges(node, marked_byte, std::forward<decltype(neighbourhood)>(neighbourhood));
  return offset;
}

std::size_t CompressedEdgesBuilder::size() const {
  return static_cast<std::size_t>(_compressed_data - _compressed_data_start.get());
}

const std::uint8_t *CompressedEdgesBuilder::compressed_data() const {
  return _compressed_data_start.get();
}

heap_profiler::unique_ptr<std::uint8_t> CompressedEdgesBuilder::take_compressed_data() {
  return std::move(_compressed_data_start);
}

std::size_t CompressedEdgesBuilder::max_degree() const {
  return _max_degree;
}

std::int64_t CompressedEdgesBuilder::total_edge_weight() const {
  return _total_edge_weight;
}

std::size_t CompressedEdgesBuilder::num_high_degree_nodes() const {
  return _num_high_degree_nodes;
}

std::size_t CompressedEdgesBuilder::num_high_degree_parts() const {
  return _num_high_degree_parts;
}

std::size_t CompressedEdgesBuilder::num_interval_nodes() const {
  return _num_interval_nodes;
}

std::size_t CompressedEdgesBuilder::num_intervals() const {
  return _num_intervals;
}

template <typename Container>
void CompressedEdgesBuilder::add_edges(
    const NodeID node, std::uint8_t *marked_byte, Container &&neighbourhood
) {
  const auto store_edge_weight = [&](const EdgeWeight edge_weight) {
    _edge_weights[_edge++] = edge_weight;
    _total_edge_weight += edge_weight;
  };

  NodeID local_degree = neighbourhood.size();

  // Find intervals [i, j] of consecutive adjacent nodes i, i + 1, ..., j - 1, j of length at least
  // kIntervalLengthTreshold. Instead of storing all nodes, only encode the left extreme i and the
  // length j - i + 1. Left extremes are stored using the differences between each left extreme and
  // the previous right extreme minus 2 (because there must be at least one integer between the end
  // of an interval and the beginning of the next one), except the first left extreme, which is
  // stored directly. The lengths are decremented by kIntervalLengthTreshold, the minimum length of
  // an interval.
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
      NodeID prev_adjacent_node = (*neighbourhood.begin()).first;

      for (auto iter = neighbourhood.begin() + 1; iter != neighbourhood.end(); ++iter) {
        const NodeID adjacent_node = (*iter).first;

        if (prev_adjacent_node + 1 == adjacent_node) {
          interval_len++;

          // The interval ends if there are no more nodes or the next node is not the increment of
          // the current node.
          if (iter + 1 == neighbourhood.end() || (*(iter + 1)).first != adjacent_node + 1) {
            if (interval_len >= CompressedGraph::kIntervalLengthTreshold) {
              const NodeID left_extreme = adjacent_node + 1 - interval_len;
              const NodeID left_extreme_gap = left_extreme + 2 - previous_right_extreme;
              const NodeID interval_length_gap =
                  interval_len - CompressedGraph::kIntervalLengthTreshold;

              _compressed_data += varint_encode(left_extreme_gap, _compressed_data);
              _compressed_data += varint_encode(interval_length_gap, _compressed_data);

              for (NodeID i = 0; i < interval_len; ++i) {
                std::pair<NodeID, EdgeWeight> &incident_edge = *(iter + 1 + i - interval_len);

                // Set the adjacent node to a special value, which indicates for the gap encoder
                // that the node has been encoded through an interval.
                incident_edge.first = std::numeric_limits<NodeID>::max();

                if (_has_edge_weights) {
                  store_edge_weight(incident_edge.second);
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

  // Store the remaining adjacent nodes using gap encoding. That is instead of directly storing the
  // nodes v_1, v_2, ..., v_{k - 1}, v_k, store the gaps v_1 - u, v_2 - v_1 - 1, ..., v_k - v_{k -
  // 1} - 1 between the nodes, where u is the source node. Note that all gaps except the first one
  // have to be positive as we sorted the nodes in ascending order. Thus, only for the first gap
  // the sign is additionally stored.
  auto iter = neighbourhood.begin();

  // Go to the first adjacent node that has not been encoded through an interval.
  if constexpr (CompressedGraph::kIntervalEncoding) {
    while ((*iter).first == std::numeric_limits<NodeID>::max()) {
      ++iter;
    }
  }

  const auto [first_adjacent_node, first_edge_weight] = *iter++;
  const SignedID first_gap = first_adjacent_node - static_cast<SignedID>(node);
  _compressed_data += signed_varint_encode(first_gap, _compressed_data);

  if (_has_edge_weights) {
    store_edge_weight(first_edge_weight);
  }

  VarIntRunLengthEncoder<NodeID> rl_encoder(_compressed_data);
  VarIntStreamEncoder<NodeID> sv_encoder(_compressed_data, local_degree - 1);

  NodeID prev_adjacent_node = first_adjacent_node;
  while (iter != neighbourhood.end()) {
    const auto [adjacent_node, edge_weight] = *iter++;

    // Skip the adjacent node since it has been encoded through an interval.
    if constexpr (CompressedGraph::kIntervalEncoding) {
      if (adjacent_node == std::numeric_limits<NodeID>::max()) {
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

    if (_has_edge_weights) {
      store_edge_weight(edge_weight);
    }

    prev_adjacent_node = adjacent_node;
  }

  if constexpr (CompressedGraph::kRunLengthEncoding) {
    rl_encoder.flush();
  } else if constexpr (CompressedGraph::kStreamEncoding) {
    sv_encoder.flush();
  }
}

CompressedGraph CompressedGraphBuilder::compress(const CSRGraph &graph) {
  const bool store_node_weights = graph.is_node_weighted();
  const bool store_edge_weights = graph.is_edge_weighted();

  CompressedGraphBuilder builder(
      graph.n(), graph.m(), store_node_weights, store_edge_weights, graph.sorted()
  );

  std::vector<std::pair<NodeID, EdgeWeight>> neighbourhood;
  neighbourhood.reserve(graph.max_degree());

  for (const NodeID node : graph.nodes()) {
    for (const auto [incident_edge, adjacent_node] : graph.neighbors(node)) {
      neighbourhood.emplace_back(adjacent_node, graph.edge_weight(incident_edge));
    }

    builder.add_node(node, neighbourhood);
    if (store_node_weights) {
      builder.add_node_weight(node, graph.node_weight(node));
    }

    neighbourhood.clear();
  }

  return builder.build();
}

CompressedGraphBuilder::CompressedGraphBuilder(
    const NodeID num_nodes,
    const EdgeID num_edges,
    const bool has_node_weights,
    const bool has_edge_weights,
    const bool sorted
)
    : _compressed_edges_builder(num_nodes, num_edges, has_edge_weights, _edge_weights) {
  KASSERT(num_nodes < std::numeric_limits<NodeID>::max() - 1);
  const std::size_t max_size = compressed_edge_array_max_size(num_nodes, num_edges);

  _nodes.resize(math::byte_width(max_size), num_nodes + 1);
  _sorted = sorted;

  _compressed_edges_builder.init(0);
  _num_edges = num_edges;

  if (has_node_weights) {
    _node_weights.resize(num_nodes);
  }

  if (has_edge_weights) {
    _edge_weights.resize(num_edges);
  }

  _store_node_weights = has_node_weights;
  _total_node_weight = 0;
}

void CompressedGraphBuilder::add_node(
    const NodeID node, std::vector<std::pair<NodeID, EdgeWeight>> &neighbourhood
) {
  KASSERT(node + 1 < _nodes.size());

  const EdgeID offset = _compressed_edges_builder.add(node, neighbourhood);
  _nodes.write(node, offset);
}

void CompressedGraphBuilder::add_node_weight(const NodeID node, const NodeWeight weight) {
  KASSERT(_store_node_weights);

  _total_node_weight += weight;
  _node_weights[node] = weight;
}

CompressedGraph CompressedGraphBuilder::build() {
  std::size_t compressed_edges_size = _compressed_edges_builder.size();
  heap_profiler::unique_ptr<std::uint8_t> wrapped_compressed_edges =
      _compressed_edges_builder.take_compressed_data();

  // Store in the last entry of the node array the offset one after the last byte belonging to the
  // last node.
  _nodes.write(_nodes.size() - 1, static_cast<EdgeID>(compressed_edges_size));

  // Store at the end of the compressed edge array the (gap of the) id of the last edge. This
  // ensures that the the degree of the last node can be computed from the difference between the
  // last two first edge ids.
  const EdgeID last_edge = _num_edges;
  std::uint8_t *compressed_edges_end = wrapped_compressed_edges.get() + compressed_edges_size;
  if constexpr (CompressedGraph::kIntervalEncoding) {
    compressed_edges_size += marked_varint_encode(last_edge, false, compressed_edges_end);
  } else {
    compressed_edges_size += varint_encode(last_edge, compressed_edges_end);
  }

  // Add an additional 15 bytes to the compressed edge array when stream encoding is enabled to
  // avoid a possible segmentation fault as the stream decoder reads 16-byte chunks.
  if constexpr (CompressedGraph::kStreamEncoding) {
    compressed_edges_size += 15;
  }

  if constexpr (kHeapProfiling) {
    heap_profiler::HeapProfiler::global().record_alloc(
        wrapped_compressed_edges.get(), compressed_edges_size
    );
  }

  RECORD("compressed_edges")
  StaticArray<std::uint8_t> compressed_edges(
      compressed_edges_size, std::move(wrapped_compressed_edges)
  );

  const bool unit_node_weights = static_cast<NodeID>(_total_node_weight + 1) == _nodes.size();
  if (unit_node_weights) {
    _node_weights.free();
  }

  const bool unit_edge_weights =
      static_cast<EdgeID>(_compressed_edges_builder.total_edge_weight()) == _num_edges;
  if (unit_edge_weights) {
    _edge_weights.free();
  }

  return CompressedGraph(
      std::move(_nodes),
      std::move(compressed_edges),
      std::move(_node_weights),
      std::move(_edge_weights),
      _num_edges,
      _compressed_edges_builder.max_degree(),
      _sorted,
      _compressed_edges_builder.num_high_degree_nodes(),
      _compressed_edges_builder.num_high_degree_parts(),
      _compressed_edges_builder.num_interval_nodes(),
      _compressed_edges_builder.num_intervals()
  );
}

std::size_t CompressedGraphBuilder::currently_used_memory() const {
  return _nodes.allocated_size() + _compressed_edges_builder.size() +
         _node_weights.size() * sizeof(NodeWeight) + _edge_weights.size() * sizeof(EdgeWeight);
}

std::int64_t CompressedGraphBuilder::total_node_weight() const {
  return _total_node_weight;
}

std::int64_t CompressedGraphBuilder::total_edge_weight() const {
  return _compressed_edges_builder.total_edge_weight();
}

CompressedGraph ParallelCompressedGraphBuilder::compress(const CSRGraph &graph) {
  const bool has_node_weights = graph.is_node_weighted();
  const bool has_edge_weights = graph.is_edge_weighted();

  ParallelCompressedGraphBuilder builder(
      graph.n(), graph.m(), has_node_weights, has_edge_weights, graph.sorted()
  );

  tbb::enumerable_thread_specific<std::vector<EdgeID>> offsets_ets;
  tbb::enumerable_thread_specific<std::vector<std::pair<NodeID, EdgeWeight>>> neighbourhood_ets;
  tbb::enumerable_thread_specific<CompressedEdgesBuilder> neighbourhood_builder_ets([&] {
    return CompressedEdgesBuilder(graph.n(), graph.m(), has_edge_weights, builder.edge_weights());
  });

  ConcurrentCircularVectorSpinlock<NodeID, EdgeID> buffer(tbb::this_task_arena::max_concurrency());

  constexpr NodeID chunk_size = 4096;
  const NodeID num_chunks = math::div_ceil(graph.n(), chunk_size);
  const NodeID last_chunk_size =
      ((graph.n() % chunk_size) != 0) ? (graph.n() % chunk_size) : chunk_size;

  tbb::parallel_for<NodeID>(0, num_chunks, [&](const auto) {
    std::vector<EdgeID> &offsets = offsets_ets.local();
    std::vector<std::pair<NodeID, EdgeWeight>> &neighbourhood = neighbourhood_ets.local();
    CompressedEdgesBuilder &neighbourhood_builder = neighbourhood_builder_ets.local();

    NodeWeight local_node_weight = 0;
    EdgeWeight local_edge_weight = 0;

    const NodeID chunk = buffer.next();
    const NodeID start_node = chunk * chunk_size;

    const NodeID chunk_length = (chunk + 1 == num_chunks) ? last_chunk_size : chunk_size;
    const NodeID end_node = start_node + chunk_length;

    const EdgeID first_edge = graph.first_edge(start_node);
    neighbourhood_builder.init(first_edge);

    for (NodeID node = start_node; node < end_node; ++node) {
      for (const auto [incident_edge, adjacent_node] : graph.neighbors(node)) {
        neighbourhood.emplace_back(adjacent_node, graph.edge_weight(incident_edge));
      }

      const EdgeID local_offset = neighbourhood_builder.add(node, neighbourhood);
      offsets.push_back(local_offset);

      neighbourhood.clear();
    }

    const EdgeID compressed_neighborhoods_size = neighbourhood_builder.size();
    const EdgeID offset = buffer.fetch_and_update(chunk, compressed_neighborhoods_size);

    NodeID node = start_node;
    for (EdgeID local_offset : offsets) {
      builder.add_node(node, offset + local_offset);

      if (has_node_weights) {
        const NodeWeight node_weight = graph.node_weight(node);
        local_node_weight += node_weight;

        builder.add_node_weight(node, node_weight);
      }

      node += 1;
    }
    offsets.clear();

    builder.add_compressed_edges(
        offset, compressed_neighborhoods_size, neighbourhood_builder.compressed_data()
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

  return builder.build();
}

ParallelCompressedGraphBuilder::ParallelCompressedGraphBuilder(
    const NodeID num_nodes,
    const EdgeID num_edges,
    const bool has_node_weights,
    const bool has_edge_weights,
    const bool sorted
) {
  KASSERT(num_nodes != std::numeric_limits<NodeID>::max() - 1);
  const std::size_t max_size = compressed_edge_array_max_size(num_nodes, num_edges);

  _nodes.resize(math::byte_width(max_size), num_nodes + 1);
  _sorted = sorted;

  _compressed_edges = heap_profiler::overcommit_memory<std::uint8_t>(max_size);
  _compressed_edges_size = 0;
  _num_edges = num_edges;

  if (has_node_weights) {
    _node_weights.resize(num_nodes);
  }

  if (has_edge_weights) {
    _edge_weights.resize(num_edges);
  }

  _max_degree = 0;
  _total_node_weight = 0;
  _total_edge_weight = 0;

  _num_high_degree_nodes = 0;
  _num_high_degree_parts = 0;
  _num_interval_nodes = 0;
  _num_intervals = 0;
}

void ParallelCompressedGraphBuilder::add_node(const NodeID node, const EdgeID offset) {
  _nodes.write(node, offset);
}

void ParallelCompressedGraphBuilder::add_compressed_edges(
    const EdgeID offset, const EdgeID length, const std::uint8_t *data
) {
  __atomic_fetch_add(&_compressed_edges_size, length, __ATOMIC_RELAXED);
  std::memcpy(_compressed_edges.get() + offset, data, length);
}

void ParallelCompressedGraphBuilder::add_node_weight(const NodeID node, const NodeWeight weight) {
  _node_weights[node] = weight;
}

void ParallelCompressedGraphBuilder::record_local_statistics(
    NodeID max_degree,
    NodeWeight node_weight,
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

  __atomic_fetch_add(&_total_node_weight, node_weight, __ATOMIC_RELAXED);
  __atomic_fetch_add(&_total_edge_weight, edge_weight, __ATOMIC_RELAXED);

  __atomic_fetch_add(&_num_high_degree_nodes, num_high_degree_nodes, __ATOMIC_RELAXED);
  __atomic_fetch_add(&_num_high_degree_parts, num_high_degree_parts, __ATOMIC_RELAXED);
  __atomic_fetch_add(&_num_interval_nodes, num_interval_nodes, __ATOMIC_RELAXED);
  __atomic_fetch_add(&_num_intervals, num_intervals, __ATOMIC_RELAXED);
}

StaticArray<EdgeWeight> &ParallelCompressedGraphBuilder::edge_weights() {
  return _edge_weights;
}

CompressedGraph ParallelCompressedGraphBuilder::build() {
  // Store in the last entry of the node array the offset one after the last byte belonging to the
  // last node.
  _nodes.write(_nodes.size() - 1, _compressed_edges_size);

  // Store at the end of the compressed edge array the (gap of the) id of the last edge. This
  // ensures that the the degree of the last node can be computed from the difference between the
  // last two first edge ids.
  std::uint8_t *_compressed_edges_end = _compressed_edges.get() + _compressed_edges_size;
  const EdgeID last_edge = _num_edges;
  if constexpr (CompressedGraph::kIntervalEncoding) {
    _compressed_edges_size += marked_varint_encode(last_edge, false, _compressed_edges_end);
  } else {
    _compressed_edges_size += varint_encode(last_edge, _compressed_edges_end);
  }

  // Add an additional 15 bytes to the compressed edge array when stream encoding is enabled to
  // avoid a possible segmentation fault as the stream decoder reads 16-byte chunks.
  if constexpr (CompressedGraph::kStreamEncoding) {
    _compressed_edges_size += 15;
  }

  if constexpr (kHeapProfiling) {
    heap_profiler::HeapProfiler::global().record_alloc(
        _compressed_edges.get(), _compressed_edges_size
    );
  }

  RECORD("compressed_edges")
  StaticArray<std::uint8_t> compressed_edges(_compressed_edges_size, std::move(_compressed_edges));

  const bool unit_node_weights = static_cast<NodeID>(_total_node_weight + 1) == _nodes.size();
  if (unit_node_weights) {
    _node_weights.free();
  }

  const bool unit_edge_weights = static_cast<EdgeID>(_total_edge_weight) == _num_edges;
  if (unit_edge_weights) {
    _edge_weights.free();
  }

  return CompressedGraph(
      std::move(_nodes),
      std::move(compressed_edges),
      std::move(_node_weights),
      std::move(_edge_weights),
      _num_edges,
      _max_degree,
      _sorted,
      _num_high_degree_nodes,
      _num_high_degree_parts,
      _num_interval_nodes,
      _num_intervals
  );
}

} // namespace kaminpar::shm
