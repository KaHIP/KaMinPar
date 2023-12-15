/*******************************************************************************
 * IO utilities for the compressed graph binary.
 *
 * @file:   shm_compressed_graph_binary.cc
 * @author: Daniel Salwasser
 * @date:   12.12.2023
 ******************************************************************************/
#include "apps/io/shm_compressed_graph_binary.h"

#include <filesystem>
#include <fstream>

#include "kaminpar-common/logger.h"

namespace kaminpar::shm::io::compressed_binary {

template <typename T> static void write_int(std::ofstream &out, const T id) {
  out.write(reinterpret_cast<const char *>(&id), sizeof(T));
}

template <typename T>
static void write_static_array(std::ofstream &out, const StaticArray<T> &static_array) {
  out.write(reinterpret_cast<const char *>(static_array.data()), static_array.size() * sizeof(T));
}

void write(const std::string &filename, const CompressedGraph &graph) {
  std::ofstream out(filename, std::ios::binary);

  write_int(out, kMagicNumber);

  write_int(out, static_cast<std::uint8_t>(sizeof(CompressedGraph::NodeID)));
  write_int(out, static_cast<std::uint8_t>(sizeof(CompressedGraph::EdgeID)));
  write_int(out, static_cast<std::uint8_t>(sizeof(CompressedGraph::NodeWeight)));
  write_int(out, static_cast<std::uint8_t>(sizeof(CompressedGraph::EdgeWeight)));

  write_int(out, CompressedGraph::kHighDegreeThreshold);
  write_int(out, static_cast<std::uint8_t>(CompressedGraph::kIntervalEncoding));
  write_int(out, CompressedGraph::kIntervalLengthTreshold);

  write_int(out, graph.n());
  write_int(out, graph.m());
  write_int(out, graph.max_degree());
  write_int(out, graph.raw_compressed_edges().size());
  write_int(out, static_cast<std::uint8_t>(graph.is_node_weighted()));
  write_int(out, static_cast<std::uint8_t>(graph.is_edge_weighted()));

  write_int(out, graph.high_degree_count());
  write_int(out, graph.part_count());
  write_int(out, graph.interval_count());

  write_static_array(out, graph.raw_nodes());
  write_static_array(out, graph.raw_compressed_edges());

  if (graph.is_node_weighted()) {
    write_static_array(out, graph.raw_node_weights());
  }

  if (graph.is_edge_weighted()) {
    write_static_array(out, graph.raw_edge_weights());
  }
}

template <typename T> static T read_int(std::ifstream &in) {
  T t;
  in.read(reinterpret_cast<char *>(&t), sizeof(T));
  return t;
}

template <typename T>
static StaticArray<T> read_static_array(std::ifstream &in, const std::size_t size) {
  T *ptr = static_cast<T *>(std::malloc(sizeof(T) * size));
  in.read(reinterpret_cast<char *>(ptr), sizeof(T) * size);
  return StaticArray<T>(ptr, size);
}

CompressedGraph read(const std::string &filename) {
  using NodeID = CompressedGraph::NodeID;
  using EdgeID = CompressedGraph::EdgeID;
  using NodeWeight = CompressedGraph::NodeWeight;
  using EdgeWeight = CompressedGraph::EdgeWeight;

  std::ifstream in(filename, std::ios::binary);

  if (kMagicNumber != read_int<std::uint64_t>(in)) {
    LOG_ERROR << "The magic number of the file is not correct!";
    std::exit(1);
  }

  std::uint8_t stored_node_id_size = read_int<std::uint8_t>(in);
  if (stored_node_id_size != sizeof(NodeID)) {
    LOG_ERROR << "The stored compressed graph uses " << (stored_node_id_size * 8)
              << "-Bit NodeIDs but this build uses " << (sizeof(NodeID) * 8) << "-Bit NodeIDs.";
    std::exit(1);
  }

  std::uint8_t stored_edge_id_size = read_int<std::uint8_t>(in);
  if (stored_edge_id_size != sizeof(EdgeID)) {
    LOG_ERROR << "The stored compressed graph uses " << (stored_edge_id_size * 8)
              << "-Bit EdgeIDs but this build uses " << (sizeof(EdgeID) * 8) << "-Bit EdgeIDs.";
    std::exit(1);
  }

  std::uint8_t stored_node_weight_size = read_int<std::uint8_t>(in);
  if (stored_node_weight_size != sizeof(NodeWeight)) {
    LOG_ERROR << "The stored compressed graph uses " << (stored_node_weight_size * 8)
              << "-Bit NodeWeights but this build uses " << (sizeof(NodeWeight) * 8)
              << "-Bit NodeWeights.";
    std::exit(1);
  }

  std::uint8_t stored_edge_weight_size = read_int<std::uint8_t>(in);
  if (stored_edge_weight_size != sizeof(EdgeWeight)) {
    LOG_ERROR << "The stored compressed graph uses " << (stored_edge_weight_size * 8)
              << "-Bit EdgeWeights but this build uses " << (sizeof(EdgeWeight) * 8)
              << "-Bit EdgeWeights.";
    std::exit(1);
  }

  NodeID high_degree_threshold = read_int<NodeID>(in);
  if (high_degree_threshold != CompressedGraph::kHighDegreeThreshold) {
    LOG_ERROR << "The stored compressed graph uses " << high_degree_threshold
              << "as the high degree threshold but this build uses "
              << (CompressedGraph::kHighDegreeThreshold) << " as the high degree threshold.";
    std::exit(1);
  }

  bool interval_encoding = static_cast<bool>(read_int<std::uint8_t>(in));
  if (interval_encoding != CompressedGraph::kIntervalEncoding) {
    if (interval_encoding) {
      LOG_ERROR << "The stored compressed graph uses interval encoding but this build does not.";
    } else {
      LOG_ERROR
          << "The stored compressed graph does not use interval encoding but this build does.";
    }
    std::exit(1);
  }

  NodeID interval_length_threshold = read_int<NodeID>(in);
  if (interval_length_threshold != CompressedGraph::kIntervalLengthTreshold) {
    LOG_ERROR << "The stored compressed graph uses " << interval_length_threshold
              << "as the interval length threshold but this build uses "
              << (CompressedGraph::kIntervalLengthTreshold) << " as the interval length threshold.";
    std::exit(1);
  }

  NodeID n = read_int<NodeID>(in);
  EdgeID m = read_int<EdgeID>(in);
  NodeID max_degree = read_int<NodeID>(in);
  std::size_t compressed_edges_size = read_int<std::size_t>(in);
  bool is_node_weighted = static_cast<bool>(read_int<std::uint8_t>(in));
  bool is_edge_weighted = static_cast<bool>(read_int<std::uint8_t>(in));

  std::size_t high_degree_count = read_int<std::size_t>(in);
  std::size_t part_count = read_int<std::size_t>(in);
  std::size_t interval_count = read_int<std::size_t>(in);

  StaticArray<EdgeID> nodes = read_static_array<EdgeID>(in, n + 1);
  StaticArray<std::uint8_t> compressed_edges =
      read_static_array<std::uint8_t>(in, compressed_edges_size);
  StaticArray<NodeWeight> node_weights =
      is_node_weighted ? read_static_array<NodeWeight>(in, n) : StaticArray<NodeWeight>();
  StaticArray<EdgeWeight> edge_weights =
      is_edge_weighted ? read_static_array<EdgeWeight>(in, m) : StaticArray<EdgeWeight>();

  return CompressedGraph(
      std::move(nodes),
      std::move(compressed_edges),
      std::move(node_weights),
      std::move(edge_weights),
      m,
      max_degree,
      high_degree_count,
      part_count,
      interval_count
  );
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
