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

ParhipHeader read_header(std::ifstream &in) {
  std::array<std::uint64_t, 3> header;
  in.read(reinterpret_cast<char *>(header.data()), kParhipHeaderSize);

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

  ParhipHeader header = read_header(in);
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

} // namespace kaminpar::shm::io::parhip
