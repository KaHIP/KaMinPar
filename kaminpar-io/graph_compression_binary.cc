/*******************************************************************************
 * IO utilities for the compressed graph binary.
 *
 * @file:   graph_compression_binary.cc
 * @author: Daniel Salwasser
 * @date:   12.12.2023
 ******************************************************************************/
#include "kaminpar-io/graph_compression_binary.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>

#include "kaminpar-shm/datastructures/compressed_graph.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::shm::io::compressed_binary {

struct CompressedBinaryHeader {
  bool has_node_weights;
  bool has_edge_weights;

  bool has_64_bit_node_id;
  bool has_64_bit_edge_id;

  bool has_64_bit_node_weight;
  bool has_64_bit_edge_weight;

  bool use_degree_bucket_order;

  bool use_high_degree_encoding;
  bool use_interval_encoding;
  bool use_streamvbyte_encoding;

  std::uint64_t high_degree_threshold;
  std::uint64_t high_degree_part_length;
  std::uint64_t interval_length_threshold;

  std::uint64_t num_nodes;
  std::uint64_t num_edges;
  std::int64_t total_edge_weight;
  std::uint64_t max_degree;

  std::uint64_t num_high_degree_nodes;
  std::uint64_t num_high_degree_parts;
  std::uint64_t num_interval_nodes;
  std::uint64_t num_intervals;
};

CompressedBinaryHeader create_header(const CompressedGraph &graph) {
  return {
      graph.is_node_weighted(),
      graph.is_edge_weighted(),

      sizeof(CompressedGraph::NodeID) == 8,
      sizeof(CompressedGraph::EdgeID) == 8,

      sizeof(CompressedGraph::NodeWeight) == 8,
      sizeof(CompressedGraph::EdgeWeight) == 8,

      graph.sorted(),

      CompressedGraph::kHighDegreeEncoding,
      CompressedGraph::kIntervalEncoding,
      CompressedGraph::kStreamVByteEncoding,

      CompressedGraph::kHighDegreeThreshold,
      CompressedGraph::kHighDegreePartLength,
      CompressedGraph::kIntervalLengthTreshold,

      graph.n(),
      graph.m(),
      graph.total_edge_weight(),
      graph.max_degree(),

      graph.num_high_degree_nodes(),
      graph.num_high_degree_parts(),
      graph.num_interval_nodes(),
      graph.num_intervals()
  };
}

template <typename T> static void write_int(std::ofstream &out, const T id) {
  out.write(reinterpret_cast<const char *>(&id), sizeof(T));
}

static void write_header(std::ofstream &out, const CompressedBinaryHeader header) {
  const std::uint16_t boolean_values =
      (header.use_streamvbyte_encoding << 9) | (header.use_interval_encoding << 8) |
      (header.use_high_degree_encoding << 7) | (header.use_degree_bucket_order << 6) |
      (header.has_64_bit_edge_weight << 5) | (header.has_64_bit_node_weight << 4) |
      (header.has_64_bit_edge_id << 3) | (header.has_64_bit_node_id << 2) |
      (header.has_edge_weights << 1) | (header.has_node_weights);
  write_int(out, boolean_values);

  write_int(out, header.high_degree_threshold);
  write_int(out, header.high_degree_part_length);
  write_int(out, header.interval_length_threshold);

  write_int(out, header.num_nodes);
  write_int(out, header.num_edges);
  write_int(out, header.total_edge_weight);
  write_int(out, header.max_degree);

  write_int(out, header.num_high_degree_nodes);
  write_int(out, header.num_high_degree_parts);
  write_int(out, header.num_interval_nodes);
  write_int(out, header.num_intervals);
}

template <typename T>
static void write_compact_static_array(std::ofstream &out, const CompactStaticArray<T> &array) {
  write_int(out, array.byte_width());
  write_int(out, array.memory_space());
  out.write(reinterpret_cast<const char *>(array.data()), array.memory_space());
}

template <typename T>
static void write_static_array(std::ofstream &out, const StaticArray<T> &static_array) {
  write_int(out, static_array.size());
  out.write(reinterpret_cast<const char *>(static_array.data()), static_array.size() * sizeof(T));
}

void write(const std::string &filename, const CompressedGraph &graph) {
  std::ofstream out(filename, std::ios::binary);
  write_int(out, kMagicNumber);

  CompressedBinaryHeader header = create_header(graph);
  write_header(out, header);

  write_compact_static_array(out, graph.raw_nodes());
  write_static_array(out, graph.raw_compressed_edges());

  if (graph.is_node_weighted()) {
    write_static_array(out, graph.raw_node_weights());
  }
}

template <typename T> static T read_int(std::ifstream &in) {
  T t;
  in.read(reinterpret_cast<char *>(&t), sizeof(T));
  return t;
}

CompressedBinaryHeader read_header(std::ifstream &in) {
  const auto boolean_values = read_int<std::uint16_t>(in);
  return {
      (boolean_values & 1) != 0,   (boolean_values & 2) != 0,   (boolean_values & 4) != 0,
      (boolean_values & 8) != 0,   (boolean_values & 16) != 0,  (boolean_values & 32) != 0,
      (boolean_values & 64) != 0,  (boolean_values & 128) != 0, (boolean_values & 256) != 0,
      (boolean_values & 512) != 0, read_int<std::uint64_t>(in), read_int<std::uint64_t>(in),
      read_int<std::uint64_t>(in), read_int<std::uint64_t>(in), read_int<std::uint64_t>(in),
      read_int<std::int64_t>(in),  read_int<std::uint64_t>(in), read_int<std::uint64_t>(in),
      read_int<std::uint64_t>(in), read_int<std::uint64_t>(in), read_int<std::uint64_t>(in),
  };
}

void verify_header(const CompressedBinaryHeader header) {
  using NodeID = CompressedGraph::NodeID;
  using EdgeID = CompressedGraph::EdgeID;
  using NodeWeight = CompressedGraph::NodeWeight;
  using EdgeWeight = CompressedGraph::EdgeWeight;

  if (header.has_64_bit_node_id) {
    if (sizeof(NodeID) != 8) {
      LOG_ERROR << "The stored compressed graph uses 64-Bit node IDs but this build uses 32-Bit "
                   "node IDs.";
      std::exit(1);
    }
  } else if (sizeof(NodeID) != 4) {
    LOG_ERROR
        << "The stored compressed graph uses 32-Bit node IDs but this build uses 64-Bit node IDs.";
    std::exit(1);
  }

  if (header.has_64_bit_edge_id) {
    if (sizeof(EdgeID) != 8) {
      LOG_ERROR << "The stored compressed graph uses 64-Bit edge IDs but this build uses 32-Bit "
                   "edge IDs.";
      std::exit(1);
    }
  } else if (sizeof(EdgeID) != 4) {
    LOG_ERROR
        << "The stored compressed graph uses 32-Bit edge IDs but this build uses 64-Bit edge IDs.";
    std::exit(1);
  }

  if (header.has_64_bit_node_weight) {
    if (sizeof(NodeWeight) != 8) {
      LOG_ERROR
          << "The stored compressed graph uses 64-Bit node weights but this build uses 32-Bit "
             "node weights.";
      std::exit(1);
    }
  } else if (sizeof(NodeWeight) != 4) {
    LOG_ERROR << "The stored compressed graph uses 32-Bit node weights but this build uses 64-Bit "
                 "node weights.";
    std::exit(1);
  }

  if (header.has_64_bit_edge_weight) {
    if (sizeof(EdgeWeight) != 8) {
      LOG_ERROR
          << "The stored compressed graph uses 64-Bit edge weights but this build uses 32-Bit "
             "edge weights.";
      std::exit(1);
    }
  } else if (sizeof(EdgeWeight) != 4) {
    LOG_ERROR << "The stored compressed graph uses 32-Bit edge weights but this build uses 64-Bit "
                 "edge weights.";
    std::exit(1);
  }

  if (header.use_high_degree_encoding != CompressedGraph::kHighDegreeEncoding) {
    if (header.use_high_degree_encoding) {
      LOG_ERROR << "The stored compressed graph uses high degree encoding but this build does not.";
    } else {
      LOG_ERROR
          << "The stored compressed graph does not use high degree encoding but this build does.";
    }
    std::exit(1);
  }

  if (header.use_interval_encoding != CompressedGraph::kIntervalEncoding) {
    if (header.use_interval_encoding) {
      LOG_ERROR << "The stored compressed graph uses interval encoding but this build does not.";
    } else {
      LOG_ERROR
          << "The stored compressed graph does not use interval encoding but this build does.";
    }
    std::exit(1);
  }

  if (header.use_streamvbyte_encoding != CompressedGraph::kStreamVByteEncoding) {
    if (header.use_streamvbyte_encoding) {
      LOG_ERROR << "The stored compressed graph uses stream encoding but this build does not.";
    } else {
      LOG_ERROR << "The stored compressed graph does not use stream encoding but this build does.";
    }
    std::exit(1);
  }

  if (header.high_degree_threshold != CompressedGraph::kHighDegreeThreshold) {
    LOG_ERROR << "The stored compressed graph uses " << header.high_degree_threshold
              << " as the high degree threshold but this build uses "
              << (CompressedGraph::kHighDegreeThreshold) << " as the high degree threshold.";
    std::exit(1);
  }

  if (header.high_degree_part_length != CompressedGraph::kHighDegreePartLength) {
    LOG_ERROR << "The stored compressed graph uses " << header.high_degree_part_length
              << " as the high degree part length but this build uses "
              << (CompressedGraph::kHighDegreePartLength) << " as the high degree part length.";
    std::exit(1);
  }

  if (header.interval_length_threshold != CompressedGraph::kIntervalLengthTreshold) {
    LOG_ERROR << "The stored compressed graph uses " << header.interval_length_threshold
              << " as the interval length threshold but this build uses "
              << (CompressedGraph::kIntervalLengthTreshold) << " as the interval length threshold.";
    std::exit(1);
  }
}

template <typename T> static CompactStaticArray<T> read_compact_static_array(std::ifstream &in) {
  const auto byte_width = read_int<std::uint8_t>(in);
  const auto allocated_size = read_int<std::size_t>(in);

  auto data = std::make_unique<std::uint8_t[]>(allocated_size);
  in.read(reinterpret_cast<char *>(data.get()), allocated_size);
  return CompactStaticArray<T>(byte_width, allocated_size, std::move(data));
}

template <typename T> static StaticArray<T> read_static_array(std::ifstream &in) {
  const auto size = read_int<std::size_t>(in);
  StaticArray<T> array(size, static_array::noinit);
  in.read(reinterpret_cast<char *>(array.data()), sizeof(T) * size);
  return array;
}

std::optional<Graph> read(const std::string &filename) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    LOG_ERROR << "Could not open file " << filename;
    return std::nullopt;
  }

  if (kMagicNumber != read_int<std::uint64_t>(in)) {
    LOG_ERROR << "The magic number of the file is not correct!";
    return std::nullopt;
  }

  CompressedBinaryHeader header = read_header(in);
  verify_header(header);

  CompactStaticArray<EdgeID> nodes = read_compact_static_array<EdgeID>(in);
  StaticArray<std::uint8_t> compressed_edges = read_static_array<std::uint8_t>(in);

  StaticArray<NodeWeight> node_weights;
  if (header.has_node_weights) {
    node_weights = read_static_array<NodeWeight>(in);
  }

  CompressedNeighborhoods<NodeID, EdgeID, EdgeWeight> compressed_neighborhoods(
      std::move(nodes),
      std::move(compressed_edges),
      header.max_degree,
      header.num_edges,
      header.has_edge_weights,
      header.total_edge_weight,
      header.num_high_degree_nodes,
      header.num_high_degree_parts,
      header.num_interval_nodes,
      header.num_intervals
  );

  return Graph(std::make_unique<CompressedGraph>(
      std::move(compressed_neighborhoods), std::move(node_weights), header.use_degree_bucket_order
  ));
}

bool is_compressed(const std::string &filename) {
  const auto size = std::filesystem::file_size(filename);
  if (size < sizeof(kMagicNumber)) {
    return false;
  }

  std::ifstream in(filename, std::ios::binary);
  return kMagicNumber == read_int<std::uint64_t>(in);
}

} // namespace kaminpar::shm::io::compressed_binary
