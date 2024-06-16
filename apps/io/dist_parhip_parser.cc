/*******************************************************************************
 * Sequential and parallel ParHiP parser for distributed compressed graphs.
 *
 * @file:   dist_parhip_parser.h
 * @author: Daniel Salwasser
 * @date:   11.05.2024
 ******************************************************************************/
#include "apps/io/dist_parhip_parser.h"

#include <cstdint>
#include <numeric>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "kaminpar-mpi/datatype.h"
#include "kaminpar-mpi/utils.h"

#include "kaminpar-dist/datastructures/distributed_compressed_graph_builder.h"
#include "kaminpar-dist/datastructures/ghost_node_mapper.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/graphutils/synchronization.h"

#include "kaminpar-common/logger.h"

namespace {

class BinaryReaderException : public std::exception {
public:
  BinaryReaderException(std::string msg) : _msg(std::move(msg)) {}

  [[nodiscard]] const char *what() const noexcept override {
    return _msg.c_str();
  }

private:
  std::string _msg;
};

class BinaryReader {
public:
  BinaryReader(const std::string &filename) {
    _file = open(filename.c_str(), O_RDONLY);
    if (_file == -1) {
      throw BinaryReaderException("Cannot read the file that stores the graph");
    }

    struct stat file_info;
    if (fstat(_file, &file_info) == -1) {
      close(_file);
      throw BinaryReaderException("Cannot determine the size of the file that stores the graph");
    }

    _length = static_cast<std::size_t>(file_info.st_size);
    _data = static_cast<std::uint8_t *>(mmap(nullptr, _length, PROT_READ, MAP_PRIVATE, _file, 0));
    if (_data == MAP_FAILED) {
      close(_file);
      throw BinaryReaderException("Cannot map the file that stores the graph");
    }
  }

  ~BinaryReader() {
    munmap(_data, _length);
    close(_file);
  }

  template <typename T> [[nodiscard]] T read(std::size_t position) const {
    return *reinterpret_cast<T *>(_data + position);
  }

  template <typename T> [[nodiscard]] T *fetch(std::size_t position) const {
    return reinterpret_cast<T *>(_data + position);
  }

private:
  int _file;
  std::size_t _length;
  std::uint8_t *_data;
};

struct ParhipHeader {
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
};

} // namespace

namespace kaminpar::dist::io::parhip {

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

template <typename Lambda>
NodeID find_node_by_edge(
    const NodeID num_nodes,
    const EdgeID num_edges,
    const EdgeID edge,
    Lambda &&fetch_adjacent_offset
) {
  if (edge == 0) {
    return 0;
  }

  std::pair<NodeID, EdgeID> low = {0, 0};
  std::pair<NodeID, EdgeID> high = {num_nodes, num_edges - 1};
  while (high.first - low.first > 1) {
    std::pair<NodeID, EdgeID> mid;
    mid.first = (low.first + high.first) / 2;
    mid.second = fetch_adjacent_offset(mid.first);

    if (mid.second < edge) {
      low = mid;
    } else {
      high = mid;
    }
  }

  return high.first;
}

} // namespace

DistributedCSRGraph csr_read(const std::string &filename, const bool sorted, const MPI_Comm comm) {
  BinaryReader reader(filename);

  const auto version = reader.read<std::uint64_t>(0);
  const auto num_nodes = reader.read<std::uint64_t>(sizeof(std::uint64_t));
  const auto num_edges = reader.read<std::uint64_t>(sizeof(std::uint64_t) * 2);
  const ParhipHeader header(version, num_nodes, num_edges);

  std::size_t position = ParhipHeader::kSize;

  const EdgeID *raw_nodes = reader.fetch<EdgeID>(position);
  position += (header.num_nodes + 1) * sizeof(EdgeID);

  const NodeID *raw_edges = reader.fetch<NodeID>(position);
  position += header.num_edges + sizeof(NodeID);

  const NodeWeight *raw_node_weights = reader.fetch<NodeWeight>(position);
  position += header.num_nodes + sizeof(NodeWeight);

  const EdgeWeight *raw_edge_weights = reader.fetch<EdgeWeight>(position);

  // Since the offsets stored in the (raw) node array of the binary are relative byte adresses
  // into the binary itself, these offsets must be mapped to the actual edge IDs.
  const EdgeID nodes_offset_base = ParhipHeader::kSize + (header.num_nodes + 1) * sizeof(EdgeID);
  const auto map_edge_offset = [&](const NodeID node) {
    return (raw_nodes[node] - nodes_offset_base) / sizeof(NodeID);
  };

  const mpi::PEID size = mpi::get_comm_size(comm);
  const mpi::PEID rank = mpi::get_comm_rank(comm);

  const auto [first_edge, last_edge] = compute_edge_range(num_edges, size, rank);

  const std::uint64_t first_node =
      find_node_by_edge(num_nodes, num_edges, first_edge, map_edge_offset);
  const std::uint64_t last_node =
      find_node_by_edge(num_nodes, num_edges, last_edge, map_edge_offset);

  const NodeID num_local_nodes = last_node - first_node;
  const EdgeID num_local_edges = map_edge_offset(last_node) - map_edge_offset(first_node);

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

  StaticArray<NodeWeight> node_weights;
  if (header.has_node_weights) {
    node_weights.resize(num_local_nodes + mapper.next_ghost_node(), static_array::noinit);

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, num_local_nodes), [&](const auto &r) {
      for (NodeID u = r.begin(); u != r.end(); ++u) {
        node_weights[u] = raw_node_weights[first_node + u];
      }
    });
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

DistributedCompressedGraph
compressed_read(const std::string &filename, const bool sorted, const MPI_Comm comm) {
  BinaryReader reader(filename);

  const auto version = reader.read<std::uint64_t>(0);
  const auto num_nodes = reader.read<std::uint64_t>(sizeof(std::uint64_t));
  const auto num_edges = reader.read<std::uint64_t>(sizeof(std::uint64_t) * 2);
  const ParhipHeader header(version, num_nodes, num_edges);

  std::size_t position = ParhipHeader::kSize;

  const EdgeID *raw_nodes = reader.fetch<EdgeID>(position);
  position += (header.num_nodes + 1) * sizeof(EdgeID);

  const NodeID *raw_edges = reader.fetch<NodeID>(position);
  position += header.num_edges + sizeof(NodeID);

  const NodeWeight *raw_node_weights = reader.fetch<NodeWeight>(position);
  position += header.num_nodes + sizeof(NodeWeight);

  const EdgeWeight *raw_edge_weights = reader.fetch<EdgeWeight>(position);

  // Since the offsets stored in the (raw) node array of the binary are relative byte adresses
  // into the binary itself, these offsets must be mapped to the actual edge IDs.
  const EdgeID nodes_offset_base = ParhipHeader::kSize + (header.num_nodes + 1) * sizeof(EdgeID);
  const auto map_edge_offset = [&](const NodeID node) {
    return (raw_nodes[node] - nodes_offset_base) / sizeof(NodeID);
  };

  const mpi::PEID size = mpi::get_comm_size(comm);
  const mpi::PEID rank = mpi::get_comm_rank(comm);

  const auto [first_edge, last_edge] = compute_edge_range(num_edges, size, rank);

  const std::uint64_t first_node =
      find_node_by_edge(num_nodes, num_edges, first_edge, map_edge_offset);
  const std::uint64_t last_node =
      find_node_by_edge(num_nodes, num_edges, last_edge, map_edge_offset);

  const NodeID num_local_nodes = last_node - first_node;
  const EdgeID num_local_edges = map_edge_offset(last_node) - map_edge_offset(first_node);

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

    builder.add_node(u - first_node, neighbourhood);
    neighbourhood.clear();
  }

  StaticArray<NodeWeight> node_weights;
  if (header.has_node_weights) {
    node_weights.resize(num_local_nodes + mapper.next_ghost_node(), static_array::noinit);

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, num_local_nodes), [&](const auto &r) {
      for (NodeID u = r.begin(); u != r.end(); ++u) {
        node_weights[u] = raw_node_weights[first_node + u];
      }
    });
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

} // namespace kaminpar::dist::io::parhip
