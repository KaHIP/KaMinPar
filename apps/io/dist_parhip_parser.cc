/*******************************************************************************
 * Sequential and parallel ParHiP parser for distributed compressed graphs.
 *
 * @file:   dist_parhip_parser.h
 * @author: Daniel Salwasser
 * @date:   11.05.2024
 ******************************************************************************/
#include "apps/io/dist_parhip_parser.h"

#include <numeric>

#include "kaminpar-mpi/datatype.h"
#include "kaminpar-mpi/utils.h"

#include "kaminpar-dist/datastructures/ghost_node_mapper.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/graphutils/synchronization.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/graph_compression/compressed_neighborhoods_builder.h"

#include "apps/io/binary_util.h"

namespace {

class ParhipHeader {
  using NodeID = kaminpar::dist::NodeID;
  using EdgeID = kaminpar::dist::EdgeID;
  using NodeWeight = kaminpar::dist::NodeWeight;
  using EdgeWeight = kaminpar::dist::EdgeWeight;

public:
  static constexpr std::uint64_t kSize = 3 * sizeof(std::uint64_t);

  bool has_edge_weights;
  bool has_node_weights;
  bool has_64_bit_edge_id;
  bool has_64_bit_node_id;
  bool has_64_bit_node_weight;
  bool has_64_bit_edge_weight;
  std::uint64_t num_nodes;
  std::uint64_t num_edges;

  ParhipHeader(std::uint64_t version, std::uint64_t num_nodes, std::uint64_t num_edges)
      : has_edge_weights((version & 1) == 0),
        has_node_weights((version & 2) == 0),
        has_64_bit_edge_id((version & 4) == 0),
        has_64_bit_node_id((version & 8) == 0),
        has_64_bit_node_weight((version & 16) == 0),
        has_64_bit_edge_weight((version & 32) == 0),
        num_nodes(num_nodes),
        num_edges(num_edges) {}

  void validate() const {
    if (has_64_bit_node_id) {
      if (sizeof(NodeID) != 8) {
        LOG_ERROR << "The stored graph uses 64-Bit node IDs but this build uses 32-Bit node IDs.";
        std::exit(1);
      }
    } else if (sizeof(NodeID) != 4) {
      LOG_ERROR << "The stored graph uses 32-Bit node IDs but this build uses 64-Bit node IDs.";
      std::exit(1);
    }

    if (has_64_bit_edge_id) {
      if (sizeof(EdgeID) != 8) {
        LOG_ERROR << "The stored graph uses 64-Bit edge IDs but this build uses 32-Bit edge IDs.";
        std::exit(1);
      }
    } else if (sizeof(EdgeID) != 4) {
      LOG_ERROR << "The stored graph uses 32-Bit edge IDs but this build uses 64-Bit edge IDs.";
      std::exit(1);
    }

    if (has_node_weights) {
      if (has_64_bit_node_weight) {
        if (sizeof(NodeWeight) != 8) {
          LOG_ERROR << "The stored graph uses 64-Bit node weights but this build uses 32-Bit node "
                       "weights.";
          std::exit(1);
        }
      } else if (sizeof(NodeWeight) != 4) {
        LOG_ERROR << "The stored graph uses 32-Bit node weights but this build uses 64-Bit node "
                     "weights.";
        std::exit(1);
      }
    }

    if (has_edge_weights) {
      if (has_64_bit_edge_weight) {
        if (sizeof(EdgeWeight) != 8) {
          LOG_ERROR << "The stored graph uses 64-Bit edge weights but this build uses 32-Bit edge "
                       "weights.";
          std::exit(1);
        }
      } else if (sizeof(EdgeWeight) != 4) {
        LOG_ERROR << "The stored graph uses 32-Bit edge weights but this build uses 64-Bit edge "
                     "weights.";
        std::exit(1);
      }
    }
  }
};

} // namespace

namespace kaminpar::dist::io::parhip {
using namespace kaminpar::io;

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

template <typename Int, typename Lambda>
NodeID find_node(const NodeID num_nodes, const Int max, const Int target, Lambda &&fetch_target) {
  if (target == 0) {
    return 0;
  }

  std::pair<NodeID, Int> low = {0, 0};
  std::pair<NodeID, Int> high = {num_nodes, max};
  while (high.first - low.first > 1) {
    std::pair<NodeID, Int> mid;
    mid.first = (low.first + high.first) / 2;
    mid.second = fetch_target(mid.first);

    if (mid.second < target) {
      low = mid;
    } else {
      high = mid;
    }
  }

  return high.first;
}

template <typename Lambda>
std::pair<std::uint64_t, std::uint64_t> find_local_nodes(
    const mpi::PEID size,
    const mpi::PEID rank,
    const GraphDistribution distribution,
    const NodeID num_nodes,
    const EdgeID num_edges,
    Lambda &&fetch_edge
) {
  switch (distribution) {
  case GraphDistribution::BALANCED_NODES: {
    return compute_chunks(num_nodes, size, rank);
  }
  case GraphDistribution::BALANCED_EDGES: {
    const auto [first_edge, last_edge] = compute_chunks(num_edges, size, rank);

    const std::uint64_t first_node =
        find_node(num_nodes, num_edges - 1, first_edge, std::forward<Lambda>(fetch_edge));
    const std::uint64_t last_node =
        find_node(num_nodes, num_edges - 1, last_edge, std::forward<Lambda>(fetch_edge));

    return std::make_pair(first_node, last_node);
  }
  case GraphDistribution::BALANCED_MEMORY_SPACE: {
    const std::size_t total_memory_space = num_nodes * sizeof(EdgeID) + num_edges * sizeof(NodeID);
    const auto [memory_space_start, memory_space_end] =
        compute_chunks(total_memory_space, size, rank);

    const auto fetch_memory_space = [&](const NodeID node) {
      const EdgeID edge = fetch_edge(node + 1);
      return node * sizeof(EdgeID) + edge * sizeof(NodeID);
    };

    const std::uint64_t first_node =
        find_node(num_nodes, total_memory_space, memory_space_start, fetch_memory_space);
    const std::uint64_t last_node =
        find_node(num_nodes, total_memory_space, memory_space_end, fetch_memory_space);

    return std::make_pair(first_node, last_node);
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
  BinaryReader reader(filename);

  const auto version = reader.read<std::uint64_t>(0);
  const auto num_nodes = reader.read<std::uint64_t>(sizeof(std::uint64_t));
  const auto num_edges = reader.read<std::uint64_t>(sizeof(std::uint64_t) * 2);
  const ParhipHeader header(version, num_nodes, num_edges);
  header.validate();

  std::size_t position = ParhipHeader::kSize;

  const EdgeID *raw_nodes = reader.fetch<EdgeID>(position);
  position += (header.num_nodes + 1) * sizeof(EdgeID);

  const NodeID *raw_edges = reader.fetch<NodeID>(position);
  position += header.num_edges * sizeof(NodeID);

  const NodeWeight *raw_node_weights = reader.fetch<NodeWeight>(position);
  if (header.has_node_weights) {
    position += header.num_nodes * sizeof(NodeWeight);
  }

  const EdgeWeight *raw_edge_weights = reader.fetch<EdgeWeight>(position);

  // Since the offsets stored in the (raw) node array of the binary are relative byte adresses
  // into the binary itself, these offsets must be mapped to the actual edge IDs.
  const EdgeID nodes_offset_base = ParhipHeader::kSize + (header.num_nodes + 1) * sizeof(EdgeID);
  const auto map_edge_offset = [&](const NodeID node) {
    return (raw_nodes[node] - nodes_offset_base) / sizeof(NodeID);
  };

  const mpi::PEID size = mpi::get_comm_size(comm);
  const mpi::PEID rank = mpi::get_comm_rank(comm);

  const auto [first_node, last_node] =
      find_local_nodes(size, rank, distribution, num_nodes, num_edges, map_edge_offset);

  const NodeID num_local_nodes = last_node - first_node;
  const EdgeID num_local_edges = map_edge_offset(last_node) - map_edge_offset(first_node);

  RECORD("node_distribution") StaticArray<GlobalNodeID> node_distribution(size + 1);
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

  RECORD("edge_distribution") StaticArray<GlobalEdgeID> edge_distribution(size + 1);
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
  RECORD("edge_weights") StaticArray<EdgeWeight> edge_weights;
  if (header.has_edge_weights) {
    edge_weights.resize(num_local_edges, static_array::noinit);
  }

  EdgeID edge = 0;
  for (NodeID u = first_node; u < last_node; ++u) {
    const NodeID node = u - first_node;
    nodes[node] = edge;

    const EdgeID offset = map_edge_offset(u);
    const EdgeID next_offset = map_edge_offset(u + 1);

    const auto degree = static_cast<NodeID>(next_offset - offset);
    for (NodeID i = 0; i < degree; ++i) {
      const EdgeID e = offset + i;

      NodeID adjacent_node = raw_edges[e];
      if (adjacent_node >= first_node && adjacent_node < last_node) {
        edges[edge] = adjacent_node - first_node;
      } else {
        edges[edge] = mapper.new_ghost_node(adjacent_node);
      }

      if (header.has_edge_weights) [[unlikely]] {
        edge_weights[edge] = raw_edge_weights[e];
      }

      edge += 1;
    }
  }
  nodes[num_local_nodes] = edge;

  RECORD("node_weights") StaticArray<NodeWeight> node_weights;
  if (header.has_node_weights) {
    node_weights.resize(num_local_nodes + mapper.next_ghost_node(), static_array::noinit);

    tbb::parallel_for(
        tbb::blocked_range<NodeID>(0, num_local_nodes),
        [&, first_node = first_node](const auto &r) {
          for (NodeID u = r.begin(); u != r.end(); ++u) {
            node_weights[u] = raw_node_weights[first_node + u];
          }
        }
    );
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

DistributedCompressedGraph compressed_read(
    const std::string &filename,
    const GraphDistribution distribution,
    const bool sorted,
    const MPI_Comm comm
) {
  BinaryReader reader(filename);

  const auto version = reader.read<std::uint64_t>(0);
  const auto num_nodes = reader.read<std::uint64_t>(sizeof(std::uint64_t));
  const auto num_edges = reader.read<std::uint64_t>(sizeof(std::uint64_t) * 2);
  const ParhipHeader header(version, num_nodes, num_edges);
  header.validate();

  std::size_t position = ParhipHeader::kSize;

  const EdgeID *raw_nodes = reader.fetch<EdgeID>(position);
  position += (header.num_nodes + 1) * sizeof(EdgeID);

  const NodeID *raw_edges = reader.fetch<NodeID>(position);
  position += header.num_edges * sizeof(NodeID);

  const NodeWeight *raw_node_weights = reader.fetch<NodeWeight>(position);
  if (header.has_node_weights) {
    position += header.num_nodes * sizeof(NodeWeight);
  }

  const EdgeWeight *raw_edge_weights = reader.fetch<EdgeWeight>(position);

  // Since the offsets stored in the (raw) node array of the binary are relative byte adresses
  // into the binary itself, these offsets must be mapped to the actual edge IDs.
  const EdgeID nodes_offset_base = ParhipHeader::kSize + (header.num_nodes + 1) * sizeof(EdgeID);
  const auto map_edge_offset = [&](const NodeID node) {
    return (raw_nodes[node] - nodes_offset_base) / sizeof(NodeID);
  };

  const mpi::PEID size = mpi::get_comm_size(comm);
  const mpi::PEID rank = mpi::get_comm_rank(comm);

  const auto [first_node, last_node] =
      find_local_nodes(size, rank, distribution, num_nodes, num_edges, map_edge_offset);

  const NodeID num_local_nodes = last_node - first_node;
  const EdgeID num_local_edges = map_edge_offset(last_node) - map_edge_offset(first_node);

  RECORD("node_distribution") StaticArray<GlobalNodeID> node_distribution(size + 1);
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

  RECORD("edge_distribution") StaticArray<GlobalEdgeID> edge_distribution(size + 1);
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

  std::vector<std::pair<NodeID, EdgeWeight>> neighbourhood;
  for (NodeID u = first_node; u < last_node; ++u) {
    const EdgeID offset = map_edge_offset(u);
    const EdgeID next_offset = map_edge_offset(u + 1);

    const auto degree = static_cast<NodeID>(next_offset - offset);
    for (NodeID i = 0; i < degree; ++i) {
      const EdgeID e = offset + i;

      NodeID adjacent_node = raw_edges[e];
      if (adjacent_node >= first_node && adjacent_node < last_node) {
        adjacent_node = adjacent_node - first_node;
      } else {
        adjacent_node = mapper.new_ghost_node(adjacent_node);
      }

      EdgeWeight edge_weight;
      if (header.has_edge_weights) [[unlikely]] {
        edge_weight = raw_edge_weights[e];
      } else {
        edge_weight = 1;
      }

      neighbourhood.emplace_back(adjacent_node, edge_weight);
    }

    builder.add(u - first_node, neighbourhood);
    neighbourhood.clear();
  }

  RECORD("node_weights") StaticArray<NodeWeight> node_weights;
  if (header.has_node_weights) {
    node_weights.resize(num_local_nodes + mapper.next_ghost_node(), static_array::noinit);

    tbb::parallel_for(
        tbb::blocked_range<NodeID>(0, num_local_nodes),
        [&, first_node = first_node](const auto &r) {
          for (NodeID u = r.begin(); u != r.end(); ++u) {
            node_weights[u] = raw_node_weights[first_node + u];
          }
        }
    );
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

} // namespace kaminpar::dist::io::parhip
