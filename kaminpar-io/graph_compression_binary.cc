/*******************************************************************************
 * IO utilities for the compressed graph binary.
 *
 * @file:   graph_compression_binary.cc
 * @author: Daniel Salwasser
 * @date:   12.12.2023
 ******************************************************************************/
#include "kaminpar-io/graph_compression_binary.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>

#include "kaminpar-io/util/binary_util.h"
#include "kaminpar-io/util/io_validation.h"

#include "kaminpar-shm/datastructures/compressed_graph.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::shm::io::compressed_binary {

namespace {

using kaminpar::io::checked_add;
using kaminpar::io::checked_mul;
using kaminpar::io::raise_if;

} // namespace

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

template <typename T>
static T read_int(const kaminpar::io::BinaryReader &reader, std::size_t &pos) {
  const T value = reader.read<T>(pos);
  pos = checked_add(pos, sizeof(T), "Compressed graph binary header is too large");
  return value;
}

CompressedBinaryHeader read_header(const kaminpar::io::BinaryReader &reader, std::size_t &pos) {
  const auto boolean_values = read_int<std::uint16_t>(reader, pos);
  raise_if(
      (boolean_values & ~std::uint16_t{0x03ff}) != 0, "Invalid compressed graph binary header flags"
  );
  return {
      (boolean_values & 1) != 0,
      (boolean_values & 2) != 0,
      (boolean_values & 4) != 0,
      (boolean_values & 8) != 0,
      (boolean_values & 16) != 0,
      (boolean_values & 32) != 0,
      (boolean_values & 64) != 0,
      (boolean_values & 128) != 0,
      (boolean_values & 256) != 0,
      (boolean_values & 512) != 0,
      read_int<std::uint64_t>(reader, pos),
      read_int<std::uint64_t>(reader, pos),
      read_int<std::uint64_t>(reader, pos),
      read_int<std::uint64_t>(reader, pos),
      read_int<std::uint64_t>(reader, pos),
      read_int<std::int64_t>(reader, pos),
      read_int<std::uint64_t>(reader, pos),
      read_int<std::uint64_t>(reader, pos),
      read_int<std::uint64_t>(reader, pos),
      read_int<std::uint64_t>(reader, pos),
      read_int<std::uint64_t>(reader, pos),
  };
}

void verify_header(const CompressedBinaryHeader header) {
  using NodeID = CompressedGraph::NodeID;
  using EdgeID = CompressedGraph::EdgeID;
  using NodeWeight = CompressedGraph::NodeWeight;
  using EdgeWeight = CompressedGraph::EdgeWeight;

  if (header.has_64_bit_node_id) {
    if (sizeof(NodeID) != 8) {
      throw kaminpar::io::IOException(
          "The stored compressed graph uses 64-Bit node IDs but this build uses 32-Bit node IDs."
      );
    }
  } else if (sizeof(NodeID) != 4) {
    throw kaminpar::io::IOException(
        "The stored compressed graph uses 32-Bit node IDs but this build uses 64-Bit node IDs."
    );
  }

  if (header.has_64_bit_edge_id) {
    if (sizeof(EdgeID) != 8) {
      throw kaminpar::io::IOException(
          "The stored compressed graph uses 64-Bit edge IDs but this build uses 32-Bit edge IDs."
      );
    }
  } else if (sizeof(EdgeID) != 4) {
    throw kaminpar::io::IOException(
        "The stored compressed graph uses 32-Bit edge IDs but this build uses 64-Bit edge IDs."
    );
  }

  if (header.has_64_bit_node_weight) {
    if (sizeof(NodeWeight) != 8) {
      throw kaminpar::io::IOException(
          "The stored compressed graph uses 64-Bit node weights but this build uses 32-Bit node "
          "weights."
      );
    }
  } else if (sizeof(NodeWeight) != 4) {
    throw kaminpar::io::IOException(
        "The stored compressed graph uses 32-Bit node weights but this build uses 64-Bit node "
        "weights."
    );
  }

  if (header.has_64_bit_edge_weight) {
    if (sizeof(EdgeWeight) != 8) {
      throw kaminpar::io::IOException(
          "The stored compressed graph uses 64-Bit edge weights but this build uses 32-Bit edge "
          "weights."
      );
    }
  } else if (sizeof(EdgeWeight) != 4) {
    throw kaminpar::io::IOException(
        "The stored compressed graph uses 32-Bit edge weights but this build uses 64-Bit edge "
        "weights."
    );
  }

  if (header.use_high_degree_encoding != CompressedGraph::kHighDegreeEncoding) {
    throw kaminpar::io::IOException("Incompatible compressed graph high degree encoding flag");
  }

  if (header.use_interval_encoding != CompressedGraph::kIntervalEncoding) {
    throw kaminpar::io::IOException("Incompatible compressed graph interval encoding flag");
  }

  if (header.use_streamvbyte_encoding != CompressedGraph::kStreamVByteEncoding) {
    throw kaminpar::io::IOException("Incompatible compressed graph stream encoding flag");
  }

  if (header.high_degree_threshold != CompressedGraph::kHighDegreeThreshold) {
    throw kaminpar::io::IOException("Incompatible compressed graph high degree threshold");
  }

  if (header.high_degree_part_length != CompressedGraph::kHighDegreePartLength) {
    throw kaminpar::io::IOException("Incompatible compressed graph high degree part length");
  }

  if (header.interval_length_threshold != CompressedGraph::kIntervalLengthTreshold) {
    throw kaminpar::io::IOException("Incompatible compressed graph interval length threshold");
  }

  raise_if(
      header.num_nodes > static_cast<std::uint64_t>(std::numeric_limits<NodeID>::max()),
      "number of nodes is too large for the node ID type"
  );
  raise_if(
      header.num_nodes == std::numeric_limits<std::uint64_t>::max(), "number of nodes is too large"
  );
  raise_if(
      header.num_edges > static_cast<std::uint64_t>(std::numeric_limits<EdgeID>::max()),
      "number of edges is too large for the edge ID type"
  );
  raise_if(header.total_edge_weight < 0, "Invalid compressed graph total edge weight");
  raise_if(header.max_degree > header.num_edges, "Invalid compressed graph maximum degree");
  raise_if(
      header.num_high_degree_nodes > header.num_nodes,
      "Invalid number of high-degree nodes in compressed graph"
  );
  raise_if(
      header.num_interval_nodes > header.num_nodes,
      "Invalid number of interval nodes in compressed graph"
  );
  raise_if(
      header.num_intervals > header.num_edges, "Invalid number of intervals in compressed graph"
  );
}

template <typename T>
static CompactStaticArray<T>
read_compact_static_array(const kaminpar::io::BinaryReader &reader, std::size_t &pos) {
  const auto byte_width = read_int<std::uint8_t>(reader, pos);
  const auto allocated_size = read_int<std::size_t>(reader, pos);
  raise_if(byte_width == 0 || byte_width > sizeof(T), "Invalid compact array byte width");
  raise_if(
      allocated_size < sizeof(T) - byte_width,
      "Invalid compact array memory space in compressed graph binary"
  );
  reader.require_available(pos, allocated_size);

  auto data = std::make_unique<std::uint8_t[]>(allocated_size);
  if (allocated_size > 0) {
    std::memcpy(data.get(), reader.fetch_raw(pos), allocated_size);
  }
  pos = checked_add(pos, allocated_size, "Compressed graph binary array is too large");
  return CompactStaticArray<T>(byte_width, allocated_size, std::move(data));
}

template <typename T>
static StaticArray<T>
read_static_array(const kaminpar::io::BinaryReader &reader, std::size_t &pos) {
  const auto size = read_int<std::size_t>(reader, pos);
  const std::size_t bytes =
      checked_mul(size, sizeof(T), "Compressed graph binary array is too large");
  reader.require_available(pos, bytes);

  StaticArray<T> array(size, static_array::noinit);
  if (bytes > 0) {
    std::memcpy(array.data(), reader.fetch_raw(pos), bytes);
  }
  pos = checked_add(pos, bytes, "Compressed graph binary array is too large");
  return array;
}

std::optional<Graph> read(const std::string &filename) {
  try {
    const kaminpar::io::BinaryReader reader(filename);
    std::size_t pos = 0;

    if (kMagicNumber != read_int<std::uint64_t>(reader, pos)) {
      return std::nullopt;
    }

    CompressedBinaryHeader header = read_header(reader, pos);
    verify_header(header);

    CompactStaticArray<EdgeID> nodes = read_compact_static_array<EdgeID>(reader, pos);
    StaticArray<std::uint8_t> compressed_edges = read_static_array<std::uint8_t>(reader, pos);
    raise_if(nodes.size() != header.num_nodes + 1, "Invalid compressed graph node offsets");
    raise_if(nodes[0] != 0, "Invalid compressed graph first node offset");
    raise_if(
        nodes[header.num_nodes] != compressed_edges.size(),
        "Invalid compressed graph final node offset"
    );
    for (NodeID u = 0; u < header.num_nodes; ++u) {
      raise_if(nodes[u] > nodes[u + 1], "Compressed graph node offsets are not monotone");
    }

    StaticArray<NodeWeight> node_weights;
    if (header.has_node_weights) {
      node_weights = read_static_array<NodeWeight>(reader, pos);
      raise_if(node_weights.size() != header.num_nodes, "Invalid compressed graph node weights");
    }
    raise_if(pos != reader.length(), "Compressed graph binary file has trailing data");

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

    return Graph(
        std::make_unique<CompressedGraph>(
            std::move(compressed_neighborhoods),
            std::move(node_weights),
            header.use_degree_bucket_order
        )
    );
  } catch (const kaminpar::io::IOException &) {
    return std::nullopt;
  }
}

bool is_compressed(const std::string &filename) {
  try {
    const kaminpar::io::BinaryReader reader(filename);
    if (reader.length() < sizeof(kMagicNumber)) {
      return false;
    }
    return kMagicNumber == reader.read<std::uint64_t>(0);
  } catch (const kaminpar::io::IOException &) {
    return false;
  }
}

} // namespace kaminpar::shm::io::compressed_binary
