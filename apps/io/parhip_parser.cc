/*******************************************************************************
 * Sequential ParHiP parser.
 *
 * @file:   parhip_parser.cc
 * @author: Daniel Salwasser
 * @date:   15.02.2024
 ******************************************************************************/
#include "apps/io/parhip_parser.h"

#include <array>
#include <cstdint>
#include <fstream>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <tbb/parallel_for.h>

#include "kaminpar-common/logger.h"

namespace kaminpar::shm::io::parhip {

constexpr std::uint64_t kParhipHeaderSize = 3 * sizeof(std::uint64_t);

struct ParhipHeader {
  bool has_edge_weights;
  bool has_node_weights;
  bool has_64_bit_edge_id;
  bool has_64_bit_node_id;
  bool has_64_bit_node_weight;
  bool has_64_bit_edge_weight;
  std::uint64_t num_nodes;
  std::uint64_t num_edges;
};

ParhipHeader parse_header(std::array<std::uint64_t, 3> header) {
  const std::uint64_t version = header[0];
  return {
      (version & 1) == 0,
      (version & 2) == 0,
      (version & 4) == 0,
      (version & 8) == 0,
      (version & 16) == 0,
      (version & 32) == 0,
      header[1],
      header[2]
  };
}

void validate_ids(ParhipHeader header) {
  if (header.has_64_bit_edge_id) {
    if (sizeof(EdgeID) != 8) {
      LOG_ERROR << "The stored graph uses 64-Bit EdgeIDs but this build uses "
                << (sizeof(EdgeID) * 8) << "-Bit EdgeIDs.";
      std::exit(1);
    }
  } else if (sizeof(EdgeID) != 4) {
    LOG_ERROR << "The stored graph uses 32-Bit EdgeIDs but this build uses " << (sizeof(EdgeID) * 8)
              << "-Bit EdgeIDs.";
    std::exit(1);
  }

  if (header.has_64_bit_node_id) {
    if (sizeof(NodeID) != 8) {
      LOG_ERROR << "The stored graph uses 64-Bit NodeIDs but this build uses "
                << (sizeof(NodeID) * 8) << "-Bit NodeIDs.";
      std::exit(1);
    }
  } else if (sizeof(NodeID) != 4) {
    LOG_ERROR << "The stored graph uses 32-Bit EdgeIDs but this build uses " << (sizeof(NodeID) * 8)
              << "-Bit NodeIDs.";
    std::exit(1);
  }

  if (header.has_64_bit_node_weight) {
    if (sizeof(NodeWeight) != 8) {
      LOG_ERROR << "The stored graph uses 64-Bit node node weights but this build uses "
                << (sizeof(NodeWeight) * 8) << "-Bit node weights.";
      std::exit(1);
    }
  } else if (sizeof(NodeWeight) != 4) {
    LOG_ERROR << "The stored graph uses 32-Bit node weights but this build uses "
              << (sizeof(NodeWeight) * 8) << "-Bit node weights.";
    std::exit(1);
  }

  if (header.has_64_bit_edge_weight) {
    if (sizeof(EdgeWeight) != 8) {
      LOG_ERROR << "The stored graph uses 64-Bit node edge weights but this build uses "
                << (sizeof(EdgeWeight) * 8) << "-Bit edge weights.";
      std::exit(1);
    }
  } else if (sizeof(NodeWeight) != 4) {
    LOG_ERROR << "The stored graph uses 32-Bit edge weights but this build uses "
              << (sizeof(EdgeWeight) * 8) << "-Bit edge weights.";
    std::exit(1);
  }
}

CSRGraph read_graph(
    std::ifstream &in,
    const std::uint64_t n,
    const std::uint64_t m,
    const bool weighted_nodes,
    const bool weighted_edges,
    const bool sorted
) {
  StaticArray<EdgeID> nodes(n + 1);
  in.read(reinterpret_cast<char *>(nodes.data()), (n + 1) * sizeof(EdgeID));

  const NodeID nodes_offset = kParhipHeaderSize + (n + 1) * sizeof(EdgeID);
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, n + 1), [&](const auto &r) {
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      nodes[u] = (nodes[u] - nodes_offset) / sizeof(NodeID);
    }
  });

  StaticArray<NodeID> edges(m);
  in.read(reinterpret_cast<char *>(edges.data()), m * sizeof(NodeID));

  StaticArray<NodeWeight> node_weights;
  if (weighted_nodes) {
    node_weights.resize(n);
    in.read(reinterpret_cast<char *>(node_weights.data()), n * sizeof(NodeWeight));
  }

  StaticArray<EdgeWeight> edge_weights;
  if (weighted_edges) {
    edge_weights.resize(m);
    in.read(reinterpret_cast<char *>(edge_weights.data()), m * sizeof(EdgeWeight));
  }

  CSRGraph graph = CSRGraph(
      std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), sorted
  );

  return graph;
}

CSRGraph csr_read(const std::string &filename, const bool sorted) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    LOG_ERROR << "Cannot read graph stored at " << filename << ".";
    std::exit(1);
  }

  std::array<std::uint64_t, 3> raw_header;
  in.read(reinterpret_cast<char *>(raw_header.data()), kParhipHeaderSize);

  ParhipHeader header = parse_header(raw_header);
  validate_ids(header);

  return read_graph(
      in,
      header.num_nodes,
      header.num_edges,
      header.has_node_weights,
      header.has_edge_weights,
      sorted
  );
}

CompressedGraph compressed_read(const std::string &filename, const bool sorted) {
  const int file = open(filename.c_str(), O_RDONLY);
  if (file < 0) {
    LOG_ERROR << "Cannot read graph stored at " << filename << ".";
    std::exit(1);
  }

  struct stat file_info {};
  if (fstat(file, &file_info) < 0) {
    LOG_ERROR << "Cannot read graph stored at " << filename << ".";
    close(file);
    std::exit(1);
  }

  const std::size_t length = static_cast<std::size_t>(file_info.st_size);

  std::uint8_t *data =
      static_cast<std::uint8_t *>(mmap(nullptr, length, PROT_READ, MAP_PRIVATE, file, 0));
  if (data == MAP_FAILED) {
    LOG_ERROR << "Cannot read graph stored at " << filename << ".";
    close(file);
    std::exit(1);
  }

  std::array<std::uint64_t, 3> raw_header;
  std::memcpy(raw_header.data(), data, kParhipHeaderSize);
  data += kParhipHeaderSize;

  const ParhipHeader header = parse_header(raw_header);
  validate_ids(header);

  CompressedGraphBuilder builder;
  builder.init(
      header.num_nodes, header.num_edges, header.has_node_weights, header.has_edge_weights, sorted
  );

  const EdgeID *nodes = reinterpret_cast<const EdgeID *>(data);
  data += (header.num_nodes + 1) * sizeof(EdgeID);

  const NodeID *edges = reinterpret_cast<const NodeID *>(data);
  data += header.num_edges + sizeof(NodeID);

  const NodeWeight *node_weights = reinterpret_cast<const NodeWeight *>(data);
  data += header.num_nodes + sizeof(NodeWeight);

  const EdgeWeight *edge_weights = reinterpret_cast<const EdgeWeight *>(data);

  const NodeID nodes_offset = kParhipHeaderSize + (header.num_nodes + 1) * sizeof(EdgeID);
  std::vector<std::pair<NodeID, EdgeWeight>> neighbourhood;
  for (NodeID u = 0; u < header.num_nodes; ++u) {
    const EdgeID offset = (nodes[u] - nodes_offset) / sizeof(NodeID);
    const EdgeID next_offset = (nodes[u + 1] - nodes_offset) / sizeof(NodeID);

    const NodeID degree = static_cast<NodeID>(next_offset - offset);
    for (NodeID i = 0; i < degree; ++i) {
      const EdgeID e = offset + i;

      const NodeID adjacent_node = edges[e];
      const EdgeWeight edge_weight = header.has_edge_weights ? edge_weights[e] : 1;

      neighbourhood.push_back(std::make_pair(adjacent_node, edge_weight));
    }

    builder.add_node(u, neighbourhood);
    if (header.has_node_weights) {
      builder.set_node_weight(u, node_weights[u]);
    }

    neighbourhood.clear();
  }

  munmap(data, length);
  close(file);
  return builder.build();
}

} // namespace kaminpar::shm::io::parhip
