/*******************************************************************************
 * Sequential and parallel ParHiP parser for distributed compressed graphs.
 *
 * @file:   dist_parhip_parser.h
 * @author: Daniel Salwasser
 * @date:   11.05.2024
 ******************************************************************************/
#include "kaminpar-io/dist_parhip_parser.h"

#include <numeric>

#include "kaminpar-io/util/binary_util.h"
#include "kaminpar-io/util/io_validation.h"

#include "kaminpar-mpi/datatype.h"
#include "kaminpar-mpi/utils.h"

#include "kaminpar-dist/datastructures/ghost_node_mapper.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/graphutils/synchronization.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/graph_compression/compressed_neighborhoods_builder.h"

namespace {

using kaminpar::io::checked_add;
using kaminpar::io::checked_cast;
using kaminpar::io::checked_mul;
using kaminpar::io::ensure_fits;
using kaminpar::io::fetch_signed;
using kaminpar::io::fetch_unsigned;
using kaminpar::io::parse_positive_weight;
using kaminpar::io::raise_if;

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
        num_edges(num_edges),
        _node_id_width(has_64_bit_node_id ? 8 : 4),
        _edge_id_width(has_64_bit_edge_id ? 8 : 4),
        _node_weight_width(has_64_bit_node_weight ? 8 : 4),
        _edge_weight_width(has_64_bit_edge_weight ? 8 : 4),
        _nodes_offset_base(checked_add(
            ParhipHeader::kSize,
            checked_mul(
                checked_add(num_nodes, 1, "ParHIP graph has too many nodes"),
                _edge_id_width,
                "ParHIP graph file layout is too large"
            ),
            "ParHIP graph file layout is too large"
        )) {}

  [[nodiscard]] std::size_t nodes_offset() const {
    return ParhipHeader::kSize;
  }

  [[nodiscard]] std::size_t edges_offset() const {
    return _nodes_offset_base;
  }

  [[nodiscard]] std::size_t node_weights_offset() const {
    return checked_add(
        edges_offset(),
        checked_mul(num_edges, _node_id_width, "ParHIP graph file layout is too large"),
        "ParHIP graph file layout is too large"
    );
  }

  [[nodiscard]] std::size_t edge_weights_offset() const {
    return checked_add(
        node_weights_offset(),
        has_node_weights
            ? checked_mul(num_nodes, _node_weight_width, "ParHIP graph file layout is too large")
            : 0,
        "ParHIP graph file layout is too large"
    );
  }

  [[nodiscard]] std::size_t file_size() const {
    return checked_add(
        edge_weights_offset(),
        has_edge_weights
            ? checked_mul(num_edges, _edge_weight_width, "ParHIP graph file layout is too large")
            : 0,
        "ParHIP graph file layout is too large"
    );
  }

  [[nodiscard]] EdgeID map_edge_offset(const std::uint64_t edge_offset) const {
    raise_if(edge_offset < _nodes_offset_base, "Invalid ParHIP node offset");
    const std::uint64_t relative_offset = edge_offset - _nodes_offset_base;
    raise_if(relative_offset % _node_id_width != 0, "Invalid ParHIP node offset alignment");
    const std::uint64_t edge = relative_offset / _node_id_width;
    raise_if(edge > num_edges, "ParHIP node offset points past the edge array");
    return checked_cast<EdgeID>(edge, "ParHIP node offset is too large for the edge ID type");
  }

  void validate(const kaminpar::io::BinaryReader &reader) const {
    ensure_fits<NodeID>(num_nodes, "number of nodes is too large for the node ID type");
    ensure_fits<EdgeID>(num_edges, "number of edges is too large for the edge ID type");
    raise_if(file_size() != reader.length(), "ParHIP graph file has an unexpected size");
  }

  [[nodiscard]] std::size_t node_id_width() const {
    return _node_id_width;
  }

  [[nodiscard]] std::size_t edge_id_width() const {
    return _edge_id_width;
  }

private:
  std::size_t _node_id_width;
  std::size_t _edge_id_width;
  std::size_t _node_weight_width;
  std::size_t _edge_weight_width;
  std::size_t _nodes_offset_base;
};

} // namespace

namespace kaminpar::dist::io::parhip {
using namespace kaminpar::io;

namespace {

template <typename FetchRawOffset>
void validate_raw_node_offsets(const ParhipHeader &header, FetchRawOffset &&fetch_raw_offset) {
  EdgeID previous = header.map_edge_offset(fetch_raw_offset(0));
  raise_if(previous != 0, "Invalid ParHIP first node offset");

  for (NodeID u = 0; u < header.num_nodes; ++u) {
    const EdgeID current = previous;
    const EdgeID next = header.map_edge_offset(fetch_raw_offset(u + 1));
    raise_if(current > next, "ParHIP node offsets are not monotone");
    previous = next;
  }

  raise_if(previous != header.num_edges, "Invalid ParHIP final node offset");
}

NodeID parse_edge_endpoint(const std::uint64_t v, const ParhipHeader &header) {
  raise_if(v >= header.num_nodes, "ParHIP edge endpoint is out of bounds");
  return checked_cast<NodeID>(v, "ParHIP edge endpoint is too large for the node ID type");
}

template <typename FetchNodeWeight, typename FetchEdgeWeight>
void collectively_validate_local_weight_sums(
    const ParhipHeader &header,
    const NodeID first_node,
    const NodeID last_node,
    const EdgeID first_edge,
    const EdgeID last_edge,
    FetchNodeWeight &&fetch_node_weight,
    FetchEdgeWeight &&fetch_edge_weight,
    const MPI_Comm comm
) {
  int local_weights_are_valid = 1;
  try {
    if (header.has_node_weights) {
      validate_weight_sum<NodeWeight>(
          last_node - first_node,
          [&](const std::size_t i) {
            return fetch_node_weight(first_node + static_cast<NodeID>(i));
          },
          "Local ParHIP node weight total is too large"
      );
    }
    if (header.has_edge_weights) {
      validate_weight_sum<EdgeWeight>(
          last_edge - first_edge,
          [&](const std::size_t i) {
            return fetch_edge_weight(first_edge + static_cast<EdgeID>(i));
          },
          "Local ParHIP edge weight total is too large"
      );
    }
  } catch (const IOException &) {
    local_weights_are_valid = 0;
  }

  int weights_are_valid = 0;
  MPI_Allreduce(&local_weights_are_valid, &weights_are_valid, 1, MPI_INT, MPI_LAND, comm);
  raise_if(weights_are_valid == 0, "Invalid ParHIP node or edge weight total");
}

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
    const std::size_t edge_id_width,
    const std::size_t node_id_width,
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
    const std::size_t total_memory_space = checked_add(
        checked_mul(num_nodes, edge_id_width, "ParHIP graph file layout is too large"),
        checked_mul(num_edges, node_id_width, "ParHIP graph file layout is too large"),
        "ParHIP graph file layout is too large"
    );
    const auto [memory_space_start, memory_space_end] =
        compute_chunks(total_memory_space, size, rank);

    const auto fetch_memory_space = [&](const NodeID node) {
      const EdgeID edge = fetch_edge(node + 1);
      return checked_add(
          checked_mul(node, edge_id_width, "ParHIP graph file layout is too large"),
          checked_mul(edge, node_id_width, "ParHIP graph file layout is too large"),
          "ParHIP graph file layout is too large"
      );
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
  raise_if((version & ~std::uint64_t{63}) != 0, "Invalid ParHIP graph file version");
  const auto num_nodes = reader.read<std::uint64_t>(sizeof(std::uint64_t));
  const auto num_edges = reader.read<std::uint64_t>(sizeof(std::uint64_t) * 2);
  const ParhipHeader header(version, num_nodes, num_edges);
  header.validate(reader);
  const NodeID n = static_cast<NodeID>(header.num_nodes);
  const EdgeID m = static_cast<EdgeID>(header.num_edges);

  const auto *raw_nodes = reader.fetch_raw(header.nodes_offset());
  validate_raw_node_offsets(header, [&](const NodeID u) {
    return fetch_unsigned(raw_nodes, header.has_64_bit_edge_id, u);
  });

  const auto *raw_edges = header.num_edges > 0 ? reader.fetch_raw(header.edges_offset()) : nullptr;

  const auto *raw_node_weights = header.has_node_weights && header.num_nodes > 0
                                     ? reader.fetch_raw(header.node_weights_offset())
                                     : nullptr;

  const auto *raw_edge_weights = header.has_edge_weights && header.num_edges > 0
                                     ? reader.fetch_raw(header.edge_weights_offset())
                                     : nullptr;

  const auto map_edge_offset = [&](const NodeID node) {
    return header.map_edge_offset(fetch_unsigned(raw_nodes, header.has_64_bit_edge_id, node));
  };
  const auto fetch_adjacent_node = [&](const EdgeID e) {
    return parse_edge_endpoint(fetch_unsigned(raw_edges, header.has_64_bit_node_id, e), header);
  };
  const auto fetch_node_weight = [&](const std::uint64_t u) {
    return parse_positive_weight<NodeWeight>(
        fetch_signed(raw_node_weights, header.has_64_bit_node_weight, u),
        "Invalid ParHIP node weight"
    );
  };
  const auto fetch_edge_weight = [&](const EdgeID e) {
    return parse_positive_weight<EdgeWeight>(
        fetch_signed(raw_edge_weights, header.has_64_bit_edge_weight, e),
        "Invalid ParHIP edge weight"
    );
  };

  const mpi::PEID size = mpi::get_comm_size(comm);
  const mpi::PEID rank = mpi::get_comm_rank(comm);

  const auto [first_node, last_node] = find_local_nodes(
      size,
      rank,
      distribution,
      n,
      m,
      header.edge_id_width(),
      header.node_id_width(),
      map_edge_offset
  );

  const NodeID num_local_nodes = last_node - first_node;
  const EdgeID first_edge = map_edge_offset(first_node);
  const EdgeID last_edge = map_edge_offset(last_node);
  const EdgeID num_local_edges = last_edge - first_edge;
  collectively_validate_local_weight_sums(
      header,
      first_node,
      last_node,
      first_edge,
      last_edge,
      fetch_node_weight,
      fetch_edge_weight,
      comm
  );

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
  StaticArray<EdgeWeight> edge_weights;
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

      NodeID adjacent_node = fetch_adjacent_node(e);
      if (adjacent_node >= first_node && adjacent_node < last_node) {
        edges[edge] = adjacent_node - first_node;
      } else {
        edges[edge] = mapper.new_ghost_node(adjacent_node);
      }

      if (header.has_edge_weights) [[unlikely]] {
        edge_weights[edge] = fetch_edge_weight(e);
      }

      edge += 1;
    }
  }
  nodes[num_local_nodes] = edge;

  StaticArray<NodeWeight> node_weights;
  if (header.has_node_weights) {
    node_weights.resize(mapper.next_ghost_node(), static_array::noinit);

    tbb::parallel_for(
        tbb::blocked_range<NodeID>(0, num_local_nodes),
        [&, first_node = first_node](const auto &r) {
          for (NodeID u = r.begin(); u != r.end(); ++u) {
            node_weights[u] = fetch_node_weight(first_node + u);
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
  raise_if((version & ~std::uint64_t{63}) != 0, "Invalid ParHIP graph file version");
  const auto num_nodes = reader.read<std::uint64_t>(sizeof(std::uint64_t));
  const auto num_edges = reader.read<std::uint64_t>(sizeof(std::uint64_t) * 2);
  const ParhipHeader header(version, num_nodes, num_edges);
  header.validate(reader);
  const NodeID n = static_cast<NodeID>(header.num_nodes);
  const EdgeID m = static_cast<EdgeID>(header.num_edges);

  const auto *raw_nodes = reader.fetch_raw(header.nodes_offset());
  validate_raw_node_offsets(header, [&](const NodeID u) {
    return fetch_unsigned(raw_nodes, header.has_64_bit_edge_id, u);
  });

  const auto *raw_edges = header.num_edges > 0 ? reader.fetch_raw(header.edges_offset()) : nullptr;

  const auto *raw_node_weights = header.has_node_weights && header.num_nodes > 0
                                     ? reader.fetch_raw(header.node_weights_offset())
                                     : nullptr;

  const auto *raw_edge_weights = header.has_edge_weights && header.num_edges > 0
                                     ? reader.fetch_raw(header.edge_weights_offset())
                                     : nullptr;

  const auto map_edge_offset = [&](const NodeID node) {
    return header.map_edge_offset(fetch_unsigned(raw_nodes, header.has_64_bit_edge_id, node));
  };
  const auto fetch_adjacent_node = [&](const EdgeID e) {
    return parse_edge_endpoint(fetch_unsigned(raw_edges, header.has_64_bit_node_id, e), header);
  };
  const auto fetch_node_weight = [&](const std::uint64_t u) {
    return parse_positive_weight<NodeWeight>(
        fetch_signed(raw_node_weights, header.has_64_bit_node_weight, u),
        "Invalid ParHIP node weight"
    );
  };
  const auto fetch_edge_weight = [&](const EdgeID e) {
    return parse_positive_weight<EdgeWeight>(
        fetch_signed(raw_edge_weights, header.has_64_bit_edge_weight, e),
        "Invalid ParHIP edge weight"
    );
  };

  const mpi::PEID size = mpi::get_comm_size(comm);
  const mpi::PEID rank = mpi::get_comm_rank(comm);

  const auto [first_node, last_node] = find_local_nodes(
      size,
      rank,
      distribution,
      n,
      m,
      header.edge_id_width(),
      header.node_id_width(),
      map_edge_offset
  );

  const NodeID num_local_nodes = last_node - first_node;
  const EdgeID first_edge = map_edge_offset(first_node);
  const EdgeID last_edge = map_edge_offset(last_node);
  const EdgeID num_local_edges = last_edge - first_edge;
  collectively_validate_local_weight_sums(
      header,
      first_node,
      last_node,
      first_edge,
      last_edge,
      fetch_node_weight,
      fetch_edge_weight,
      comm
  );

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

  std::vector<std::pair<NodeID, EdgeWeight>> neighbourhood;
  for (NodeID u = first_node; u < last_node; ++u) {
    const EdgeID offset = map_edge_offset(u);
    const EdgeID next_offset = map_edge_offset(u + 1);

    const auto degree = static_cast<NodeID>(next_offset - offset);
    for (NodeID i = 0; i < degree; ++i) {
      const EdgeID e = offset + i;

      NodeID adjacent_node = fetch_adjacent_node(e);
      if (adjacent_node >= first_node && adjacent_node < last_node) {
        adjacent_node = adjacent_node - first_node;
      } else {
        adjacent_node = mapper.new_ghost_node(adjacent_node);
      }

      EdgeWeight edge_weight;
      if (header.has_edge_weights) [[unlikely]] {
        edge_weight = fetch_edge_weight(e);
      } else {
        edge_weight = 1;
      }

      neighbourhood.emplace_back(adjacent_node, edge_weight);
    }

    builder.add(u - first_node, std::span<std::pair<NodeID, EdgeWeight>>(neighbourhood));
    neighbourhood.clear();
  }

  StaticArray<NodeWeight> node_weights;
  if (header.has_node_weights) {
    node_weights.resize(mapper.next_ghost_node(), static_array::noinit);

    tbb::parallel_for(
        tbb::blocked_range<NodeID>(0, num_local_nodes),
        [&, first_node = first_node](const auto &r) {
          for (NodeID u = r.begin(); u != r.end(); ++u) {
            node_weights[u] = fetch_node_weight(first_node + u);
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
