/*******************************************************************************
 * Sequential METIS parser for distributed graphs.
 *
 * @file:   dist_metis_parser.h
 * @author: Daniel Salwasser
 * @date:   22.06.2024
 ******************************************************************************/
#include "apps/io/dist_metis_parser.h"

#include <numeric>

#include "kaminpar-mpi/datatype.h"
#include "kaminpar-mpi/utils.h"

#include "kaminpar-dist/datastructures/ghost_node_mapper.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/graphutils/synchronization.h"

#include "kaminpar-common/graph_compression/compressed_neighborhoods_builder.h"

#include "apps/io/file_toker.h"

namespace kaminpar::dist::io::metis {
using namespace kaminpar::io;

namespace {

struct MetisHeader {
  std::uint64_t num_nodes = 0;
  std::uint64_t num_edges = 0;
  bool has_node_weights = false;
  bool has_edge_weights = false;
};

MetisHeader parse_header(MappedFileToker &toker) {
  toker.skip_spaces();
  while (toker.current() == '%') {
    toker.skip_line();
    toker.skip_spaces();
  }

  const std::uint64_t num_nodes = toker.scan_uint();
  const std::uint64_t num_edges = toker.scan_uint() * 2;
  const std::uint64_t format = (toker.current() != '\n') ? toker.scan_uint() : 0;
  toker.consume_char('\n');

  if (format != 0 && format != 1 && format != 10 && format != 11 && format && format != 100 &&
      format != 110 && format != 101 && format != 111) {
    LOG_WARNING << "invalid or unsupported graph format";
  }

  [[maybe_unused]] const bool has_node_sizes = format / 100; // == 1xx
  const bool has_node_weights = (format % 100) / 10;         // == x1x
  const bool has_edge_weights = format % 10;                 // == xx1

  if (has_node_sizes) {
    LOG_WARNING << "ignoring node sizes";
  }

  KASSERT(
      num_nodes <= static_cast<std::uint64_t>(std::numeric_limits<NodeID>::max()),
      "number of nodes is too large for the node ID type"
  );
  KASSERT(
      num_edges <= static_cast<std::uint64_t>(std::numeric_limits<EdgeID>::max()),
      "number of edges is too large for the edge ID type"
  );
  KASSERT(
      num_edges <= (num_nodes * (num_nodes - 1)) / 2,
      "specified number of edges is impossibly large"
  );

  return {
      .num_nodes = num_nodes,
      .num_edges = num_edges,
      .has_node_weights = has_node_weights,
      .has_edge_weights = has_edge_weights,
  };
}

template <typename NextNodeCB, typename NextEdgeCB>
void parse_graph(
    MappedFileToker &toker,
    const MetisHeader header,
    NextNodeCB &&next_node_cb,
    NextEdgeCB &&next_edge_cb
) {
  static_assert(std::is_invocable_v<NextNodeCB, std::uint64_t>);
  static_assert(std::is_invocable_v<NextEdgeCB, std::uint64_t, std::uint64_t>);
  constexpr bool stoppable = std::is_invocable_r_v<bool, NextNodeCB, std::uint64_t>;

  for (std::uint64_t u = 0; u < header.num_nodes; ++u) {
    toker.skip_spaces();
    while (toker.current() == '%') {
      toker.skip_line();
      toker.skip_spaces();
    }

    std::uint64_t node_weight = 1;
    if (header.has_node_weights) {
      node_weight = toker.scan_uint();
    }

    if constexpr (stoppable) {
      if (next_node_cb(node_weight)) {
        return;
      }
    } else {
      next_node_cb(node_weight);
    }

    while (std::isdigit(toker.current())) {
      const std::uint64_t v = toker.scan_uint() - 1;

      std::uint64_t edge_weight = 1;
      if (header.has_edge_weights) {
        edge_weight = toker.scan_uint();
      }

      next_edge_cb(edge_weight, v);
    }

    if (toker.valid_position()) {
      toker.consume_char('\n');
    }
  }
}

} // namespace

namespace {

template <typename Int>
std::pair<Int, Int>
compute_chunks(const Int length, const mpi::PEID num_processes, const mpi::PEID rank) {
  const Int chunk_size = length / num_processes;
  const Int remainder = length % num_processes;
  const Int from = rank * chunk_size + std::min<Int>(rank, remainder);
  const Int to = std::min<Int>(
      from + ((static_cast<Int>(rank) < remainder) ? chunk_size + 1 : chunk_size), length
  );
  return std::make_pair(from, to);
}

std::tuple<NodeID, NodeID, EdgeID, std::size_t> find_node_by_node(
    MappedFileToker &toker,
    const MetisHeader header,
    const EdgeID first_node,
    const EdgeID last_node
) {
  std::size_t start_pos = 0;
  EdgeID actual_first_edge = 0;

  NodeID current_node = 0;
  EdgeID current_edge = 0;
  parse_graph(
      toker,
      header,
      [&](const auto) {
        if (current_node < first_node) {
          current_node += 1;
          return false;
        }

        if (current_node < last_node) {
          if (current_node - first_node == 0) {
            start_pos = toker.position();
            actual_first_edge = current_edge;
          }

          current_node += 1;
          return false;
        }

        return true;
      },
      [&](const auto, const auto) { current_edge += 1; }
  );

  const EdgeID num_edges = ((last_node - first_node) == 0) ? 0 : current_edge - actual_first_edge;
  return std::make_tuple(first_node, last_node, num_edges, start_pos);
}

std::tuple<NodeID, NodeID, EdgeID, std::size_t> find_node_by_edge(
    MappedFileToker &toker,
    const MetisHeader header,
    const EdgeID first_edge,
    const EdgeID last_edge
) {
  NodeID first_node = 0;
  NodeID length = 0;

  std::size_t start_pos = 0;
  EdgeID actual_first_edge = 0;

  EdgeID current_edge = 0;
  parse_graph(
      toker,
      header,
      [&](const auto) {
        if (current_edge < first_edge) {
          first_node += 1;
          return false;
        }

        if (current_edge < last_edge) {
          if (length == 0) {
            start_pos = toker.position();
            actual_first_edge = current_edge;
          }

          length += 1;
          return false;
        }

        return true;
      },
      [&](const auto, const auto) { current_edge += 1; }
  );

  const EdgeID num_edges = (length == 0) ? 0 : current_edge - actual_first_edge;
  return std::make_tuple(first_node, first_node + length, num_edges, start_pos);
}

std::tuple<NodeID, NodeID, EdgeID, std::size_t> find_node_by_memory_space(
    MappedFileToker &toker,
    const MetisHeader header,
    const std::size_t memory_space_start,
    const std::size_t memory_space_stop
) {
  NodeID first_node = 0;
  NodeID length = 0;

  std::size_t start_pos = 0;
  EdgeID first_edge = 0;

  EdgeID current_edge = 0;
  parse_graph(
      toker,
      header,
      [&](const auto) {
        std::size_t memory_space = first_node * sizeof(EdgeID) + current_edge * sizeof(NodeID);
        if (memory_space < memory_space_start) {
          first_node += 1;
          return false;
        }

        memory_space += length * sizeof(EdgeID);
        if (memory_space < memory_space_stop) {
          if (length == 0) {
            start_pos = toker.position();
            first_edge = current_edge;
          }

          length += 1;
          return false;
        }

        return true;
      },
      [&](const auto, const auto) { current_edge += 1; }
  );

  const EdgeID num_edges = (length == 0) ? 0 : current_edge - first_edge;
  return std::make_tuple(first_node, first_node + length, num_edges, start_pos);
}

std::tuple<NodeID, NodeID, EdgeID, std::size_t> find_local_nodes(
    const mpi::PEID size,
    const mpi::PEID rank,
    MappedFileToker &toker,
    const MetisHeader header,
    const GraphDistribution distribution
) {
  switch (distribution) {
  case GraphDistribution::BALANCED_NODES: {
    const auto [first_node, last_node] = compute_chunks(header.num_nodes, size, rank);
    return find_node_by_node(toker, header, first_node, last_node);
  }
  case GraphDistribution::BALANCED_EDGES: {
    const auto [first_edge, last_edge] = compute_chunks(header.num_edges, size, rank);
    return find_node_by_edge(toker, header, first_edge, last_edge);
  }
  case GraphDistribution::BALANCED_MEMORY_SPACE: {
    const std::size_t total_memory_space =
        header.num_nodes * sizeof(EdgeID) + header.num_edges * sizeof(NodeID);
    const auto [memory_space_start, memory_space_end] =
        compute_chunks(total_memory_space, size, rank);

    return find_node_by_memory_space(toker, header, memory_space_start, memory_space_end);
  }
  default:
    __builtin_unreachable();
  }
}

} // namespace

DistributedCSRGraph csr_read(
    const std::string &filename,
    const GraphDistribution distribution,
    const bool sorted,
    const MPI_Comm comm
) {
  MappedFileToker toker(filename);
  MetisHeader header = parse_header(toker);

  const mpi::PEID size = mpi::get_comm_size(comm);
  const mpi::PEID rank = mpi::get_comm_rank(comm);

  const auto [first_node, last_node, num_local_edges, start_pos] =
      find_local_nodes(size, rank, toker, header, distribution);
  const NodeID num_local_nodes = last_node - first_node;

  StaticArray<GlobalNodeID> node_distribution(size + 1);
  node_distribution[rank + 1] = last_node;
  MPI_Allgather(
      MPI_IN_PLACE,
      0,
      MPI_DATATYPE_NULL,
      node_distribution.data() + 1,
      1,
      mpi::type::get<GlobalNodeID>(),
      comm
  );

  StaticArray<GlobalEdgeID> edge_distribution(size + 1);
  edge_distribution[rank] = num_local_edges;
  MPI_Allgather(
      MPI_IN_PLACE,
      1,
      mpi::type::get<GlobalEdgeID>(),
      edge_distribution.data(),
      1,
      mpi::type::get<GlobalEdgeID>(),
      comm
  );
  std::exclusive_scan(
      edge_distribution.begin(),
      edge_distribution.end(),
      edge_distribution.begin(),
      static_cast<GlobalEdgeID>(0)
  );

  graph::GhostNodeMapper mapper(rank, node_distribution);
  RECORD("nodes") StaticArray<EdgeID> nodes(num_local_nodes + 1, static_array::noinit);
  RECORD("edges") StaticArray<NodeID> edges(num_local_edges, static_array::noinit);

  RECORD("node_weights") StaticArray<NodeWeight> node_weights;
  if (header.has_node_weights) {
    node_weights.resize(num_local_nodes, static_array::noinit);
  }

  RECORD("edge_weights") StaticArray<EdgeWeight> edge_weights;
  if (header.has_edge_weights) {
    edge_weights.resize(num_local_edges, static_array::noinit);
  }

  NodeID node = 0;
  EdgeID edge = 0;
  if (num_local_nodes > 0) {
    toker.seek(start_pos);
    header.num_nodes = num_local_nodes;

    parse_graph(
        toker,
        header,
        [&](const auto weight) {
          nodes[node] = edge;

          if (header.has_node_weights) {
            node_weights[node] = static_cast<NodeWeight>(weight);
          }

          node += 1;
        },
        [&, first_node = first_node, last_node = last_node](const auto weight, const auto v) {
          NodeID adjacent_node = static_cast<NodeID>(v);
          if (adjacent_node >= first_node && adjacent_node < last_node) {
            adjacent_node = adjacent_node - first_node;
          } else {
            adjacent_node = mapper.new_ghost_node(adjacent_node);
          }

          edges[edge] = adjacent_node;
          if (header.has_edge_weights) {
            edge_weights[edge] = static_cast<EdgeWeight>(weight);
          }

          edge += 1;
        }
    );
  }
  nodes[node] = edge;

  if (header.has_node_weights && mapper.next_ghost_node() > 0) {
    StaticArray<NodeWeight> actual_node_weights(
        num_local_nodes + mapper.next_ghost_node(), static_array::noinit
    );

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, num_local_nodes), [&](const auto &r) {
      for (NodeID u = r.begin(); u != r.end(); ++u) {
        actual_node_weights[u] = node_weights[u];
      }
    });

    node_weights = std::move(actual_node_weights);
  }

  auto [global_to_ghost, ghost_to_global, ghost_owner] = mapper.finalize();

  DistributedCSRGraph graph(
      std::move(node_distribution),
      std::move(edge_distribution),
      std::move(nodes),
      std::move(edges),
      std::move(node_weights),
      std::move(edge_weights),
      std::move(ghost_owner),
      std::move(ghost_to_global),
      std::move(global_to_ghost),
      sorted,
      comm
  );

  // Fill in ghost node weights
  if (header.has_node_weights) {
    graph::synchronize_ghost_node_weights(graph);
  }

  return graph;
}

DistributedCompressedGraph compress_read(
    const std::string &filename,
    const GraphDistribution distribution,
    const bool sorted,
    const MPI_Comm comm
) {
  MappedFileToker toker(filename);
  MetisHeader header = parse_header(toker);

  const mpi::PEID size = mpi::get_comm_size(comm);
  const mpi::PEID rank = mpi::get_comm_rank(comm);

  const auto [first_node, last_node, num_local_edges, start_pos] =
      find_local_nodes(size, rank, toker, header, distribution);
  const NodeID num_local_nodes = last_node - first_node;

  StaticArray<GlobalNodeID> node_distribution(size + 1);
  node_distribution[rank + 1] = last_node;
  MPI_Allgather(
      MPI_IN_PLACE,
      0,
      MPI_DATATYPE_NULL,
      node_distribution.data() + 1,
      1,
      mpi::type::get<GlobalNodeID>(),
      comm
  );

  StaticArray<GlobalEdgeID> edge_distribution(size + 1);
  edge_distribution[rank] = num_local_edges;
  MPI_Allgather(
      MPI_IN_PLACE,
      1,
      mpi::type::get<GlobalEdgeID>(),
      edge_distribution.data(),
      1,
      mpi::type::get<GlobalEdgeID>(),
      comm
  );
  std::exclusive_scan(
      edge_distribution.begin(),
      edge_distribution.end(),
      edge_distribution.begin(),
      static_cast<GlobalEdgeID>(0)
  );

  CompactGhostNodeMappingBuilder mapper(rank, node_distribution);
  CompressedNeighborhoodsBuilder<NodeID, EdgeID, EdgeWeight> builder(
      num_local_nodes, num_local_edges, header.has_edge_weights
  );

  StaticArray<NodeWeight> node_weights;
  if (header.has_node_weights) {
    node_weights.resize(num_local_edges, static_array::noinit);
  }

  if (num_local_nodes > 0) {
    toker.seek(start_pos);
    header.num_nodes = num_local_nodes;

    std::vector<std::pair<NodeID, EdgeWeight>> neighbourhood;
    NodeID node = 0;
    EdgeID edge = 0;
    parse_graph(
        toker,
        header,
        [&](const auto weight) {
          if (node > 0) {
            builder.add(node - 1, neighbourhood);
            neighbourhood.clear();
          }

          if (header.has_node_weights) {
            node_weights[node] = static_cast<NodeWeight>(weight);
          }

          node += 1;
        },
        [&, first_node = first_node, last_node = last_node](const auto weight, const auto v) {
          NodeID adjacent_node = static_cast<NodeID>(v);
          if (adjacent_node >= first_node && adjacent_node < last_node) {
            adjacent_node = adjacent_node - first_node;
          } else {
            adjacent_node = mapper.new_ghost_node(adjacent_node);
          }

          neighbourhood.emplace_back(adjacent_node, static_cast<EdgeWeight>(weight));
          edge += 1;
        }
    );

    builder.add(node - 1, neighbourhood);
    neighbourhood.clear();
    neighbourhood.shrink_to_fit();
  }

  if (header.has_node_weights && mapper.next_ghost_node() > 0) {
    StaticArray<NodeWeight> actual_node_weights(
        num_local_nodes + mapper.next_ghost_node(), static_array::noinit
    );

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, num_local_nodes), [&](const auto &r) {
      for (NodeID u = r.begin(); u != r.end(); ++u) {
        actual_node_weights[u] = node_weights[u];
      }
    });

    node_weights = std::move(actual_node_weights);
  }

  DistributedCompressedGraph graph(
      std::move(node_distribution),
      std::move(edge_distribution),
      builder.build(),
      std::move(node_weights),
      mapper.finalize(),
      sorted,
      comm
  );

  // Fill in ghost node weights
  if (header.has_node_weights) {
    graph::synchronize_ghost_node_weights(graph);
  }

  return graph;
}

} // namespace kaminpar::dist::io::metis
