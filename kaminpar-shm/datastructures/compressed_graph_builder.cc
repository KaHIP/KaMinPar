/*******************************************************************************
 * Sequential and parallel builder for compressed graphs.
 *
 * @file:   compressed_graph_builder.cc
 * @author: Daniel Salwasser
 * @date:   03.05.2024
 ******************************************************************************/
#include "kaminpar-shm/datastructures/compressed_graph_builder.h"

#include <algorithm>
#include <cstdint>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include "kaminpar-shm/kaminpar.h"

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
    const NodeID num_nodes, const EdgeID num_edges, bool has_edge_weights
)
    : _has_edge_weights(has_edge_weights) {
  const std::size_t max_size = compressed_edge_array_max_size(num_nodes, num_edges);
  _compressed_data_start = heap_profiler::overcommit_memory<std::uint8_t>(max_size);
  _compressed_data = _compressed_data_start.get();
  _compressed_data_max_size = 0;
}

CompressedEdgesBuilder::CompressedEdgesBuilder(
    const NodeID num_nodes, const EdgeID num_edges, const NodeID max_degree, bool has_edge_weights
)
    : _has_edge_weights(has_edge_weights) {
  const std::size_t max_size = compressed_edge_array_max_size<false>(num_nodes, max_degree);
  _compressed_data_start = heap_profiler::overcommit_memory<std::uint8_t>(max_size);
  _compressed_data = _compressed_data_start.get();
  _compressed_data_max_size = 0;
}

CompressedEdgesBuilder::~CompressedEdgesBuilder() {
  if constexpr (kHeapProfiling) {
    if (_compressed_data_start) {
      const auto prev_compressed_data_size =
          static_cast<std::size_t>(_compressed_data - _compressed_data_start.get());
      const std::size_t compressed_data_size =
          std::max(_compressed_data_max_size, prev_compressed_data_size);

      heap_profiler::HeapProfiler::global().record_alloc(
          _compressed_data_start.get(), compressed_data_size
      );
    }
  }
}

void CompressedEdgesBuilder::init(const EdgeID first_edge) {
  const auto prev_compressed_data_size =
      static_cast<std::size_t>(_compressed_data - _compressed_data_start.get());
  _compressed_data_max_size = std::max(_compressed_data_max_size, prev_compressed_data_size);
  _compressed_data = _compressed_data_start.get();

  _edge = first_edge;
  _max_degree = 0;
  _total_edge_weight = 0;

  _num_high_degree_nodes = 0;
  _num_high_degree_parts = 0;
  _num_interval_nodes = 0;
  _num_intervals = 0;
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

CompressedGraph CompressedGraphBuilder::compress(const CSRGraph &graph) {
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

    builder.add_node(u, neighbourhood);
    if (store_node_weights) {
      builder.add_node_weight(u, graph.node_weight(u));
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
    : _compressed_edges_builder(num_nodes, num_edges, has_edge_weights),
      _store_edge_weights(has_edge_weights) {
  KASSERT(num_nodes < std::numeric_limits<NodeID>::max() - 1);
  const std::size_t max_size = compressed_edge_array_max_size(num_nodes, num_edges);

  _nodes.resize(math::byte_width(max_size), num_nodes + 1);
  _sorted = sorted;

  _compressed_edges_builder.init(0);
  _num_edges = num_edges;

  if (has_node_weights) {
    _node_weights.resize(num_nodes);
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

  return CompressedGraph(
      std::move(_nodes),
      std::move(compressed_edges),
      std::move(_node_weights),
      _num_edges,
      _compressed_edges_builder.total_edge_weight(),
      _store_edge_weights,
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
         _node_weights.size() * sizeof(NodeWeight);
}

std::int64_t CompressedGraphBuilder::total_node_weight() const {
  return _total_node_weight;
}

std::int64_t CompressedGraphBuilder::total_edge_weight() const {
  return _compressed_edges_builder.total_edge_weight();
}

CompressedGraph ParallelCompressedGraphBuilder::compress(const CSRGraph &graph) {
  return ParallelCompressedGraphBuilder::compress(
      graph.n(),
      graph.m(),
      graph.is_node_weighted(),
      graph.is_edge_weighted(),
      graph.sorted(),
      [](const NodeID u) { return u; },
      [&](const NodeID u) { return graph.degree(u); },
      [&](const NodeID u) { return graph.first_edge(u); },
      [&](const EdgeID e) { return graph.edge_target(e); },
      [&](const NodeID u) { return graph.node_weight(u); },
      [&](const EdgeID e) { return graph.edge_weight(e); }
  );
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
  _has_edge_weights = has_edge_weights;

  if (has_node_weights) {
    _node_weights.resize(num_nodes, static_array::noinit);
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

  return CompressedGraph(
      std::move(_nodes),
      std::move(compressed_edges),
      std::move(_node_weights),
      _num_edges,
      _total_edge_weight,
      _has_edge_weights,
      _max_degree,
      _sorted,
      _num_high_degree_nodes,
      _num_high_degree_parts,
      _num_interval_nodes,
      _num_intervals
  );
}

} // namespace kaminpar::shm
