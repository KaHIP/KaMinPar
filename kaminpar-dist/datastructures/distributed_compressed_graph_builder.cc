/*******************************************************************************
 * Sequential builder for distributed compressed graphs.
 *
 * @file:   distributed_compressed_graph_builder.h
 * @author: Daniel Salwasser
 * @date:   07.06.2024
 ******************************************************************************/
#include "kaminpar-dist/datastructures/distributed_compressed_graph_builder.h"

#include "kaminpar-dist/datastructures/ghost_node_mapper.h"
#include "kaminpar-dist/graphutils/synchronization.h"

#include "kaminpar-common/assert.h"

namespace kaminpar::dist {

DistributedCompressedGraph
DistributedCompressedGraphBuilder::compress(const DistributedCSRGraph &graph) {
  const mpi::PEID size = mpi::get_comm_size(graph.communicator());
  const mpi::PEID rank = mpi::get_comm_rank(graph.communicator());

  StaticArray<GlobalNodeID> node_distribution(
      graph.node_distribution().begin(), graph.node_distribution().end()
  );
  StaticArray<GlobalEdgeID> edge_distribution(
      graph.edge_distribution().begin(), graph.edge_distribution().end()
  );

  graph::GhostNodeMapper mapper(rank, node_distribution);
  DistributedCompressedGraphBuilder builder(
      graph.n(), graph.m(), graph.is_node_weighted(), graph.is_edge_weighted(), graph.sorted()
  );

  const NodeID first_node = node_distribution[rank];
  const NodeID last_node = node_distribution[rank + 1];

  const auto &raw_nodes = graph.raw_nodes();
  const auto &raw_edges = graph.raw_nodes();
  const auto &raw_node_weights = graph.raw_nodes();

  std::vector<std::pair<NodeID, EdgeWeight>> neighbourhood;
  for (const NodeID u : graph.nodes()) {
    graph.neighbors(u, [&](const EdgeID e, const NodeID adjacent_node) {
      const EdgeWeight edge_weight = graph.is_edge_weighted() ? graph.edge_weight(e) : 1;

      if (graph.is_owned_node(adjacent_node)) {
        neighbourhood.emplace_back(adjacent_node, edge_weight);
      } else {
        const NodeID original_adjacent_node = graph.local_to_global_node(adjacent_node);
        neighbourhood.emplace_back(mapper.new_ghost_node(original_adjacent_node), edge_weight);
      }
    });

    builder.add_node(u, neighbourhood);
    neighbourhood.clear();
  }

  StaticArray<NodeWeight> node_weights;
  if (graph.is_node_weighted()) {
    node_weights.resize(graph.n() + mapper.next_ghost_node(), static_array::noinit);

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, graph.n()), [&](const auto &r) {
      for (NodeID u = r.begin(); u != r.end(); ++u) {
        node_weights[u] = raw_node_weights[first_node + u];
      }
    });
  }

  auto [global_to_ghost, ghost_to_global, ghost_owner] = mapper.finalize();
  auto [nodes, edges, edge_weights] = builder.build();

  DistributedCompressedGraph compressed_graph(
      std::move(node_distribution),
      std::move(edge_distribution),
      std::move(nodes),
      std::move(edges),
      std::move(node_weights),
      std::move(edge_weights),
      std::move(ghost_owner),
      std::move(ghost_to_global),
      std::move(global_to_ghost),
      graph.sorted(),
      graph.communicator()
  );
  return compressed_graph;
}

DistributedCompressedGraphBuilder::DistributedCompressedGraphBuilder(
    const NodeID num_nodes,
    const EdgeID num_edges,
    const bool has_node_weights,
    const bool has_edge_weights,
    const bool sorted
)
    : _compressed_edges_builder(num_nodes, num_edges, has_edge_weights, _edge_weights) {
  _sorted = sorted;
  _nodes.resize(num_nodes + 1, static_array::noinit);

  _num_edges = num_edges;
  _compressed_edges_builder.init(0);

  if (has_edge_weights) {
    _edge_weights.resize(num_edges, static_array::noinit);
  }
}

void DistributedCompressedGraphBuilder::add_node(
    const NodeID node, std::vector<std::pair<NodeID, EdgeWeight>> &neighbourhood
) {
  KASSERT(node + 1 < _nodes.size());

  const EdgeID offset = _compressed_edges_builder.add(node, neighbourhood);
  _nodes[node] = offset;
}

std::tuple<StaticArray<EdgeID>, CompressedEdges<NodeID, EdgeID>, StaticArray<EdgeWeight>>
DistributedCompressedGraphBuilder::build() {
  std::size_t compressed_edges_size = _compressed_edges_builder.size();
  heap_profiler::unique_ptr<std::uint8_t> wrapped_compressed_edges =
      _compressed_edges_builder.take_compressed_data();

  // Store in the last entry of the node array the offset one after the last byte belonging to the
  // last node.
  _nodes[_nodes.size() - 1] = static_cast<EdgeID>(compressed_edges_size);

  // Store at the end of the compressed edge array the (gap of the) id of the last edge. This
  // ensures that the the degree of the last node can be computed from the difference between
  // the last two first edge ids.
  const EdgeID last_edge = _num_edges;
  std::uint8_t *compressed_edges_end = wrapped_compressed_edges.get() + compressed_edges_size;
  if constexpr (CompressedEdges<NodeID, EdgeID>::kIntervalEncoding) {
    compressed_edges_size += marked_varint_encode(last_edge, false, compressed_edges_end);
  } else {
    compressed_edges_size += varint_encode(last_edge, compressed_edges_end);
  }

  // Add an additional 15 bytes to the compressed edge array when stream encoding is enabled to
  // avoid a possible segmentation fault as the stream decoder reads 16-byte chunks.
  if constexpr (CompressedEdges<NodeID, EdgeID>::kStreamEncoding) {
    compressed_edges_size += 15;
  }

  if constexpr (kHeapProfiling) {
    heap_profiler::HeapProfiler::global().record_alloc(
        wrapped_compressed_edges.get(), compressed_edges_size
    );
  }

  StaticArray<std::uint8_t> raw_compressed_edges(
      compressed_edges_size, std::move(wrapped_compressed_edges)
  );
  CompressedEdges<NodeID, EdgeID> compressed_edges(_num_edges, std::move(raw_compressed_edges));

  return std::make_tuple(std::move(_nodes), std::move(compressed_edges), std::move(_edge_weights));
}

} // namespace kaminpar::dist
