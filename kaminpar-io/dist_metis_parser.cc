/*******************************************************************************
 * Sequential METIS parser for distributed graphs.
 *
 * @file:   dist_metis_parser.h
 * @author: Daniel Salwasser
 * @date:   22.06.2024
 ******************************************************************************/
#include "kaminpar-io/dist_metis_parser.h"

#include <limits>
#include <numeric>

#include "kaminpar-io/util/file_toker.h"

#include "kaminpar-mpi/datatype.h"
#include "kaminpar-mpi/utils.h"

#include "kaminpar-dist/datastructures/ghost_node_mapper.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/graphutils/synchronization.h"

#include "kaminpar-common/graph_compression/compressed_neighborhoods_builder.h"

namespace kaminpar::dist::io::metis {

using namespace kaminpar::io;

namespace {

struct MetisHeader {
  std::uint64_t num_nodes = 0;
  std::uint64_t num_global_nodes = 0;
  std::uint64_t node_offset = 0;
  std::uint64_t num_edges = 0;
  bool has_node_sizes = false;
  bool has_node_weights = false;
  bool has_edge_weights = false;
};

void raise_if(const bool condition, const char *message) {
  if (condition) [[unlikely]] {
    throw IOException(message);
  }
}

template <typename Int> void ensure_fits(const std::uint64_t value, const char *message) {
  raise_if(value > static_cast<std::uint64_t>(std::numeric_limits<Int>::max()), message);
}

bool is_valid_metis_format(const std::uint64_t format) {
  return format == 0 || format == 1 || format == 10 || format == 11 || format == 100 ||
         format == 101 || format == 110 || format == 111;
}

bool is_line_end(const MappedFileToker &toker) {
  return !toker.valid_position() || toker.current() == '\n' || toker.current() == '\r';
}

void consume_line_end(MappedFileToker &toker) {
  if (!toker.valid_position()) {
    return;
  }
  if (toker.current() == '\r') {
    toker.advance();
  }
  if (toker.valid_position()) {
    toker.consume_char('\n');
  }
}

void skip_comment_and_empty_lines(MappedFileToker &toker) {
  while (toker.valid_position()) {
    toker.skip_spaces();
    if (!toker.valid_position()) {
      return;
    }
    if (toker.current() == '%' || toker.current() == '#') {
      toker.skip_line();
    } else if (is_line_end(toker)) {
      consume_line_end(toker);
    } else {
      return;
    }
  }
}

void skip_comment_lines(MappedFileToker &toker) {
  while (toker.valid_position()) {
    toker.skip_spaces();
    if (toker.valid_position() && (toker.current() == '%' || toker.current() == '#')) {
      toker.skip_line();
    } else {
      return;
    }
  }
}

void finish_line(MappedFileToker &toker) {
  toker.skip_spaces();
  raise_if(!is_line_end(toker), "Unexpected character in METIS graph file");
  consume_line_end(toker);
}

MetisHeader parse_header(MappedFileToker &toker) {
  skip_comment_and_empty_lines(toker);

  const std::uint64_t num_nodes = toker.scan_uint();
  const std::uint64_t undirected_edges = toker.scan_uint();
  raise_if(
      undirected_edges > std::numeric_limits<std::uint64_t>::max() / 2,
      "the number of adjacency entries is too large"
  );
  const std::uint64_t num_edges = undirected_edges * 2;
  const std::uint64_t format = !is_line_end(toker) ? toker.scan_uint() : 0;
  finish_line(toker);

  raise_if(!is_valid_metis_format(format), "Invalid or unsupported METIS graph format");

  const bool has_node_sizes = format / 100;          // == 1xx
  const bool has_node_weights = (format % 100) / 10; // == x1x
  const bool has_edge_weights = format % 10;         // == xx1

  if (has_node_sizes) {
    LOG_WARNING << "ignoring node sizes";
  }

  ensure_fits<NodeID>(num_nodes, "number of nodes is too large for the node ID type");
  raise_if(num_nodes == std::numeric_limits<std::uint64_t>::max(), "number of nodes is too large");
  ensure_fits<EdgeID>(num_edges, "number of edges is too large for the edge ID type");
  raise_if(
      static_cast<unsigned __int128>(undirected_edges) >
          (static_cast<unsigned __int128>(num_nodes) * (num_nodes - 1)) / 2,
      "specified number of edges is impossibly large"
  );

  return {
      .num_nodes = num_nodes,
      .num_global_nodes = num_nodes,
      .node_offset = 0,
      .num_edges = num_edges,
      .has_node_sizes = has_node_sizes,
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
  static_assert(
      std::is_invocable_v<NextNodeCB, std::uint64_t> ||
      std::is_invocable_v<NextNodeCB, std::uint64_t, std::size_t>
  );
  static_assert(std::is_invocable_v<NextEdgeCB, std::uint64_t, std::uint64_t>);

  const auto invoke_next_node = [&](const std::uint64_t node_weight,
                                    const std::size_t node_start_pos) {
    if constexpr (std::is_invocable_r_v<bool, NextNodeCB, std::uint64_t, std::size_t>) {
      return next_node_cb(node_weight, node_start_pos);
    } else if constexpr (std::is_invocable_v<NextNodeCB, std::uint64_t, std::size_t>) {
      next_node_cb(node_weight, node_start_pos);
      return false;
    } else if constexpr (std::is_invocable_r_v<bool, NextNodeCB, std::uint64_t>) {
      return next_node_cb(node_weight);
    } else {
      next_node_cb(node_weight);
      return false;
    }
  };

  for (std::uint64_t u = 0; u < header.num_nodes; ++u) {
    skip_comment_lines(toker);
    raise_if(
        !toker.valid_position(), "input contains fewer vertex lines than specified in the header"
    );

    const std::size_t node_start_pos = toker.position();

    if (header.has_node_sizes) {
      toker.skip_uint();
    }

    std::uint64_t node_weight = 1;
    if (header.has_node_weights) {
      node_weight = toker.scan_uint();
      ensure_fits<NodeWeight>(node_weight, "node weight is too large for the node weight type");
      raise_if(node_weight == 0u, "zero node weights are not supported");
    }

    if (invoke_next_node(node_weight, node_start_pos)) {
      return;
    }

    while (toker.current_is_digit()) {
      const std::uint64_t raw_v = toker.scan_uint();
      raise_if(raw_v == 0, "METIS vertex IDs must be one-based");
      const std::uint64_t v = raw_v - 1;

      std::uint64_t edge_weight = 1;
      if (header.has_edge_weights) {
        edge_weight = toker.scan_uint();
        ensure_fits<EdgeWeight>(edge_weight, "edge weight is too large for the edge weight type");
        raise_if(edge_weight == 0u, "zero edge weights are not supported");
      }

      raise_if(v >= header.num_global_nodes, "neighbor out of bounds");
      raise_if(header.node_offset + u == v, "detected illegal self-loop");
      next_edge_cb(edge_weight, v);
    }

    finish_line(toker);
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
      [&](const auto, const std::size_t node_start_pos) {
        if (current_node < first_node) {
          current_node += 1;
          return false;
        }

        if (current_node < last_node) {
          if (current_node - first_node == 0) {
            start_pos = node_start_pos;
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
      [&](const auto, const std::size_t node_start_pos) {
        if (current_edge < first_edge) {
          first_node += 1;
          return false;
        }

        if (current_edge < last_edge) {
          if (length == 0) {
            start_pos = node_start_pos;
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
      [&](const auto, const std::size_t node_start_pos) {
        std::size_t memory_space = first_node * sizeof(EdgeID) + current_edge * sizeof(NodeID);
        if (memory_space < memory_space_start) {
          first_node += 1;
          return false;
        }

        memory_space += length * sizeof(EdgeID);
        if (memory_space < memory_space_stop) {
          if (length == 0) {
            start_pos = node_start_pos;
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
  StaticArray<EdgeID> nodes(num_local_nodes + 1, static_array::noinit);
  StaticArray<NodeID> edges(num_local_edges, static_array::noinit);

  StaticArray<NodeWeight> node_weights;
  if (header.has_node_weights) {
    node_weights.resize(num_local_nodes, static_array::noinit);
  }

  StaticArray<EdgeWeight> edge_weights;
  if (header.has_edge_weights) {
    edge_weights.resize(num_local_edges, static_array::noinit);
  }

  NodeID node = 0;
  EdgeID edge = 0;
  if (num_local_nodes > 0) {
    toker.seek(start_pos);
    header.num_nodes = num_local_nodes;
    header.node_offset = first_node;

    parse_graph(
        toker,
        header,
        [&](const auto weight) {
          raise_if(node >= num_local_nodes, "input contains more vertex lines than expected");
          nodes[node] = edge;

          if (header.has_node_weights) {
            node_weights[node] = static_cast<NodeWeight>(weight);
          }

          node += 1;
        },
        [&, first_node = first_node, last_node = last_node](const auto weight, const auto v) {
          raise_if(edge >= num_local_edges, "input contains more adjacency entries than expected");
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
  raise_if(node != num_local_nodes, "input contains fewer vertex lines than expected");
  raise_if(edge != num_local_edges, "input contains fewer adjacency entries than expected");
  nodes[node] = edge;

  const NodeID num_total_nodes = mapper.next_ghost_node();
  if (header.has_node_weights && num_total_nodes > num_local_nodes) {
    StaticArray<NodeWeight> actual_node_weights(num_total_nodes, static_array::noinit);

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
    node_weights.resize(num_local_nodes, static_array::noinit);
  }

  if (num_local_nodes > 0) {
    toker.seek(start_pos);
    header.num_nodes = num_local_nodes;
    header.node_offset = first_node;

    std::vector<std::pair<NodeID, EdgeWeight>> neighbourhood;
    NodeID node = 0;
    EdgeID edge = 0;
    parse_graph(
        toker,
        header,
        [&](const auto weight) {
          raise_if(node >= num_local_nodes, "input contains more vertex lines than expected");
          if (node > 0) {
            builder.add(node - 1, std::span<std::pair<NodeID, EdgeWeight>>(neighbourhood));
            neighbourhood.clear();
          }

          if (header.has_node_weights) {
            node_weights[node] = static_cast<NodeWeight>(weight);
          }

          node += 1;
        },
        [&, first_node = first_node, last_node = last_node](const auto weight, const auto v) {
          raise_if(edge >= num_local_edges, "input contains more adjacency entries than expected");
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

    raise_if(node != num_local_nodes, "input contains fewer vertex lines than expected");
    raise_if(edge != num_local_edges, "input contains fewer adjacency entries than expected");
    builder.add(node - 1, std::span<std::pair<NodeID, EdgeWeight>>(neighbourhood));
    neighbourhood.clear();
    neighbourhood.shrink_to_fit();
  }

  const NodeID num_total_nodes = mapper.next_ghost_node();
  if (header.has_node_weights && num_total_nodes > num_local_nodes) {
    StaticArray<NodeWeight> actual_node_weights(num_total_nodes, static_array::noinit);

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
