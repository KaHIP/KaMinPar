/*******************************************************************************
 * Sequential and parallel ParHiP parser.
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
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>
#include <unistd.h>

#include "kaminpar-shm/datastructures/compressed_graph_builder.h"

#include "kaminpar-common/datastructures/concurrent_circular_vector.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::shm::io::parhip {

namespace {

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

void validate_header(const ParhipHeader header) {
  if (header.has_64_bit_node_id) {
    if (sizeof(NodeID) != 8) {
      LOG_ERROR << "The stored graph uses 64-Bit node IDs but this build uses 32-Bit node IDs.";
      std::exit(1);
    }
  } else if (sizeof(NodeID) != 4) {
    LOG_ERROR << "The stored graph uses 32-Bit node IDs but this build uses 64-Bit node IDs.";
    std::exit(1);
  }

  if (header.has_64_bit_edge_id) {
    if (sizeof(EdgeID) != 8) {
      LOG_ERROR << "The stored graph uses 64-Bit edge IDs but this build uses 32-Bit edge IDs.";
      std::exit(1);
    }
  } else if (sizeof(EdgeID) != 4) {
    LOG_ERROR << "The stored graph uses 32-Bit edge IDs but this build uses 64-Bit edge IDs.";
    std::exit(1);
  }

  if (header.has_64_bit_node_weight) {
    if (sizeof(NodeWeight) != 8) {
      LOG_ERROR << "The stored graph uses 64-Bit node weights but this build uses 32-Bit node"
                   "weights.";
      std::exit(1);
    }
  } else if (sizeof(NodeWeight) != 4) {
    LOG_ERROR << "The stored graph uses 32-Bit node weights but this build uses 64-Bit node"
                 "weights.";
    std::exit(1);
  }

  if (header.has_64_bit_edge_weight) {
    if (sizeof(EdgeWeight) != 8) {
      LOG_ERROR << "The stored graph uses 64-Bit edge weights but this build uses 32-Bit edge "
                   "weights.";
      std::exit(1);
    }
  } else if (sizeof(EdgeWeight) != 4) {
    LOG_ERROR << "The stored graph uses 32-Bit edge weights but this build uses 64-Bit edge"
                 "weights.";
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
  StaticArray<EdgeID> nodes(n + 1, static_array::noinit);
  in.read(reinterpret_cast<char *>(nodes.data()), (n + 1) * sizeof(EdgeID));

  const EdgeID nodes_offset = ParhipHeader::kSize + (n + 1) * sizeof(EdgeID);
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, n + 1), [&](const auto &r) {
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      nodes[u] = (nodes[u] - nodes_offset) / sizeof(NodeID);
    }
  });

  StaticArray<NodeID> edges(m, static_array::noinit);
  in.read(reinterpret_cast<char *>(edges.data()), m * sizeof(NodeID));

  StaticArray<NodeWeight> node_weights;
  if (weighted_nodes) {
    node_weights.resize(n, static_array::noinit);
    in.read(reinterpret_cast<char *>(node_weights.data()), n * sizeof(NodeWeight));
  }

  StaticArray<EdgeWeight> edge_weights;
  if (weighted_edges) {
    edge_weights.resize(m, static_array::noinit);
    in.read(reinterpret_cast<char *>(edge_weights.data()), m * sizeof(EdgeWeight));
  }

  return CSRGraph(
      std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), sorted
  );
}

} // namespace

CSRGraph csr_read(const std::string &filename, const bool sorted) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    LOG_ERROR << "Cannot read graph stored at " << filename << ".";
    std::exit(1);
  }

  std::array<std::uint64_t, 3> raw_header;
  in.read(reinterpret_cast<char *>(raw_header.data()), ParhipHeader::kSize);

  const ParhipHeader header(raw_header[0], raw_header[1], raw_header[2]);
  validate_header(header);

  return read_graph(
      in,
      header.num_nodes,
      header.num_edges,
      header.has_node_weights,
      header.has_edge_weights,
      sorted
  );
}

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

} // namespace

CompressedGraph compressed_read(const std::string &filename, const bool sorted) {
  try {
    BinaryReader reader(filename);

    const auto version = reader.read<std::uint64_t>(0);
    const auto num_nodes = reader.read<std::uint64_t>(sizeof(std::uint64_t));
    const auto num_edges = reader.read<std::uint64_t>(sizeof(std::uint64_t) * 2);
    const ParhipHeader header(version, num_nodes, num_edges);
    validate_header(header);

    CompressedGraphBuilder builder(
        header.num_nodes, header.num_edges, header.has_node_weights, header.has_edge_weights, sorted
    );

    std::size_t position = ParhipHeader::kSize;

    const EdgeID *nodes = reader.fetch<EdgeID>(position);
    position += (header.num_nodes + 1) * sizeof(EdgeID);

    const NodeID *edges = reader.fetch<NodeID>(position);
    position += header.num_edges + sizeof(NodeID);

    const NodeWeight *node_weights = reader.fetch<NodeWeight>(position);
    position += header.num_nodes + sizeof(NodeWeight);

    const EdgeWeight *edge_weights = reader.fetch<EdgeWeight>(position);

    // Since the offsets stored in the (raw) node array of the binary are relative byte adresses
    // into the binary itself, these offsets must be mapped to the actual edge IDs.
    const EdgeID nodes_offset_base = ParhipHeader::kSize + (header.num_nodes + 1) * sizeof(EdgeID);
    const auto map_edge_offset = [&](const NodeID node) {
      return (nodes[node] - nodes_offset_base) / sizeof(NodeID);
    };

    std::vector<std::pair<NodeID, EdgeWeight>> neighbourhood;
    for (NodeID u = 0; u < header.num_nodes; ++u) {
      const EdgeID offset = map_edge_offset(u);
      const EdgeID next_offset = map_edge_offset(u + 1);

      const auto degree = static_cast<NodeID>(next_offset - offset);
      for (NodeID i = 0; i < degree; ++i) {
        const EdgeID e = offset + i;

        const NodeID adjacent_node = edges[e];
        const EdgeWeight edge_weight = header.has_edge_weights ? edge_weights[e] : 1;

        neighbourhood.emplace_back(adjacent_node, edge_weight);
      }

      builder.add_node(u, neighbourhood);
      if (header.has_node_weights) {
        builder.add_node_weight(u, node_weights[u]);
      }

      neighbourhood.clear();
    }

    return builder.build();
  } catch (const BinaryReaderException &e) {
    LOG_ERROR << e.what();
    std::exit(1);
  }
}

CompressedGraph compressed_read_parallel(const std::string &filename, const bool sorted) {
  try {
    BinaryReader reader(filename);

    // Read information about the graph from the header and validates whether the graph can be
    // processed.
    const auto version = reader.read<std::uint64_t>(0);
    const auto num_nodes = reader.read<std::uint64_t>(sizeof(std::uint64_t));
    const auto num_edges = reader.read<std::uint64_t>(sizeof(std::uint64_t) * 2);
    const ParhipHeader header(version, num_nodes, num_edges);
    validate_header(header);

    // Initializes pointers into the binary which point to the positions where the different parts
    // of the graph are stored.
    std::size_t position = ParhipHeader::kSize;

    const EdgeID *nodes = reader.fetch<EdgeID>(position);
    position += (header.num_nodes + 1) * sizeof(EdgeID);

    const NodeID *edges = reader.fetch<NodeID>(position);
    position += header.num_edges + sizeof(NodeID);

    const NodeWeight *node_weights = reader.fetch<NodeWeight>(position);
    position += header.num_nodes + sizeof(NodeWeight);

    const EdgeWeight *edge_weights = reader.fetch<EdgeWeight>(position);

    // Since the offsets stored in the (raw) node array of the binary are relative byte adresses
    // into the binary itself, these offsets must be mapped to the actual edge IDs.
    const EdgeID nodes_offset_base = ParhipHeader::kSize + (header.num_nodes + 1) * sizeof(EdgeID);
    const auto map_edge_offset = [&](const NodeID node) {
      return (nodes[node] - nodes_offset_base) / sizeof(NodeID);
    };

    // Initializes the data structures used to build the compressed graph in parallel.
    ParallelCompressedGraphBuilder builder(
        header.num_nodes, header.num_edges, header.has_node_weights, header.has_edge_weights, sorted
    );

    tbb::enumerable_thread_specific<std::vector<EdgeID>> offsets_ets;
    tbb::enumerable_thread_specific<std::vector<std::pair<NodeID, EdgeWeight>>> neighbourhood_ets;
    tbb::enumerable_thread_specific<CompressedEdgesBuilder> neighbourhood_builder_ets([&] {
      return CompressedEdgesBuilder(
          num_nodes, num_edges, header.has_edge_weights, builder.edge_weights()
      );
    });

    ConcurrentCircularVector<NodeID, EdgeID> buffer(tbb::this_task_arena::max_concurrency());

    // To compress the graph in parallel the nodes are split into chunks. Each parallel task fetches
    // a chunk and compresses the neighbourhoods of the corresponding nodes. The compressed
    // neighborhoods are thereby temporarily stored in a buffer. They are moved into the compressed
    // edge array when the (total) length of the compressed neighborhoods of the previous chunks is
    // determined.
    constexpr NodeID chunk_size = 4096;
    const NodeID num_chunks = math::div_ceil(num_nodes, chunk_size);
    const NodeID last_chunk_size =
        ((num_nodes % chunk_size) != 0) ? (num_nodes % chunk_size) : chunk_size;

    tbb::parallel_for<NodeID>(0, num_chunks, [&](const auto) {
      std::vector<EdgeID> &offsets = offsets_ets.local();
      std::vector<std::pair<NodeID, EdgeWeight>> &neighbourhood = neighbourhood_ets.local();
      CompressedEdgesBuilder &neighbourhood_builder = neighbourhood_builder_ets.local();

      const NodeID chunk = buffer.next();
      const NodeID start_node = chunk * chunk_size;

      const NodeID chunk_length = (chunk + 1 == num_chunks) ? last_chunk_size : chunk_size;
      const NodeID end_node = start_node + chunk_length;

      EdgeID edge = map_edge_offset(start_node);
      neighbourhood_builder.init(edge);

      NodeWeight local_node_weight = 0;

      // Compress the neighborhoods of the nodes in the fetched chunk.
      for (NodeID node = start_node; node < end_node; ++node) {
        const auto degree = static_cast<NodeID>((nodes[node + 1] - nodes[node]) / sizeof(NodeID));

        for (NodeID i = 0; i < degree; ++i) {
          const NodeID adjacent_node = edges[edge];
          const EdgeWeight edge_weight = header.has_edge_weights ? edge_weights[edge] : 1;

          neighbourhood.emplace_back(adjacent_node, edge_weight);
          edge += 1;
        }

        const EdgeID local_offset = neighbourhood_builder.add(node, neighbourhood);
        offsets.push_back(local_offset);

        neighbourhood.clear();
      }

      // Wait for the parallel tasks that process the previous chunks to finish.
      const EdgeID compressed_neighborhoods_size = neighbourhood_builder.size();
      const EdgeID offset = buffer.fetch_and_update(chunk, compressed_neighborhoods_size);

      // Store the edge offset and node weight for each node in the chunk and copy the compressed
      // neighborhoods into the actual compressed edge array.
      NodeID node = start_node;
      for (EdgeID local_offset : offsets) {
        builder.add_node(node, offset + local_offset);

        if (header.has_node_weights) {
          const NodeWeight node_weight = node_weights[node];
          local_node_weight += node_weight;

          builder.add_node_weight(node, node_weight);
        }

        node += 1;
      }
      offsets.clear();

      builder.add_compressed_edges(
          offset, compressed_neighborhoods_size, neighbourhood_builder.compressed_data()
      );

      builder.record_local_statistics(
          neighbourhood_builder.max_degree(),
          local_node_weight,
          neighbourhood_builder.total_edge_weight(),
          neighbourhood_builder.num_high_degree_nodes(),
          neighbourhood_builder.num_high_degree_parts(),
          neighbourhood_builder.num_interval_nodes(),
          neighbourhood_builder.num_intervals()
      );
    });

    return builder.build();
  } catch (const BinaryReaderException &e) {
    LOG_ERROR << e.what();
    std::exit(1);
  }
}

} // namespace kaminpar::shm::io::parhip