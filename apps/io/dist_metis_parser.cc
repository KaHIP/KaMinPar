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

#include "kaminpar-dist/datastructures/distributed_compressed_graph_builder.h"
#include "kaminpar-dist/datastructures/ghost_node_mapper.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/graphutils/synchronization.h"

#include "apps/io/file_tokener.h"

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

std::pair<EdgeID, EdgeID>
compute_edge_range(const EdgeID num_edges, const mpi::PEID size, const mpi::PEID rank) {
  const EdgeID chunk = num_edges / size;
  const EdgeID rem = num_edges % size;
  const EdgeID from = rank * chunk + std::min<EdgeID>(rank, rem);
  const EdgeID to =
      std::min<EdgeID>(from + ((static_cast<EdgeID>(rank) < rem) ? chunk + 1 : chunk), num_edges);
  return std::make_pair(from, to);
}

std::tuple<NodeID, NodeID, EdgeID, std::size_t> find_node_by_edge(
    MappedFileToker &toker,
    const MetisHeader header,
    const EdgeID first_edge,
    const EdgeID last_edge
) {
  NodeID a = 0;
  NodeID first_node = 0;
  NodeID last_node = 0;
  EdgeID actual_first_edge = 0;
  std::size_t start_pos;

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
          if (last_node == 0) {
            start_pos = toker.position();
            actual_first_edge = current_edge;
          }

          last_node += 1;
          return false;
        }

        return true;
      },
      [&](const auto, const auto) { current_edge += 1; }
  );

  const EdgeID num_edges = current_edge - actual_first_edge;
  return std::make_tuple(first_node, first_node + last_node, num_edges, start_pos);
}

} // namespace

DistributedCompressedGraph
compress_read(const std::string &filename, const bool sorted, const MPI_Comm comm) {
  MappedFileToker toker(filename);
  MetisHeader header = parse_header(toker);

  const mpi::PEID size = mpi::get_comm_size(comm);
  const mpi::PEID rank = mpi::get_comm_rank(comm);

  const auto [first_edge, last_edge] = compute_edge_range(header.num_edges, size, rank);
  const auto [first_node, last_node, num_local_edges, start_pos] =
      find_node_by_edge(toker, header, first_edge, last_edge);
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
  DistributedCompressedGraphBuilder builder(
      num_local_nodes, num_local_edges, header.has_node_weights, header.has_edge_weights, sorted
  );

  StaticArray<NodeWeight> node_weights;
  if (header.has_node_weights) {
    node_weights.resize(header.num_nodes, static_array::noinit);
  }

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
          builder.add_node(node - 1, neighbourhood);
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

  builder.add_node(node - 1, neighbourhood);
  neighbourhood.clear();
  neighbourhood.shrink_to_fit();

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
  auto [nodes, edges, edge_weights] = builder.build();

  DistributedCompressedGraph graph(
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

} // namespace kaminpar::dist::io::metis
