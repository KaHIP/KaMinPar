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
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/concurrent_circular_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/loops.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::io::parhip {
SET_DEBUG(false);

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

namespace {
namespace debug {
using Duration = std::chrono::high_resolution_clock::duration;

struct Stats {
  Duration compression_time{0};
  Duration sync_time{0};
  Duration copy_time{0};

  std::size_t num_chunks{0};
  std::size_t num_edges{0};
};

template <typename Lambda> decltype(auto) scoped_time(auto &elapsed, Lambda &&l) {
  constexpr bool kNonReturning = std::is_void_v<std::invoke_result_t<Lambda>>;

  if constexpr (kDebug) {
    if constexpr (kNonReturning) {
      auto start = std::chrono::high_resolution_clock::now();
      l();
      auto end = std::chrono::high_resolution_clock::now();
      elapsed += end - start;
    } else {
      auto start = std::chrono::high_resolution_clock::now();
      decltype(auto) val = l();
      auto end = std::chrono::high_resolution_clock::now();
      elapsed += end - start;
      return val;
    }
  } else {
    return l();
  }
}

void print_stats(const auto &stats_ets) {
  DBG << "Chunk distribution:";

  std::size_t cur_thread = 0;
  for (const auto &stats : stats_ets) {
    DBG << "t" << ++cur_thread << ": " << stats.num_chunks;
  }

  DBG << "Edge distribution:";

  cur_thread = 0;
  for (const auto &stats : stats_ets) {
    DBG << "t" << ++cur_thread << ": " << stats.num_edges;
  }

  DBG << "Time distribution: (compression, sync, copy) [s]";

  const auto to_sec = [&](auto elapsed) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() / 1000.0;
  };

  Duration total_time_compression(0);
  Duration total_time_sync(0);
  Duration total_time_copy(0);

  cur_thread = 0;
  for (const auto &stats : stats_ets) {
    total_time_compression += stats.compression_time;
    total_time_sync += stats.sync_time;
    total_time_copy += stats.copy_time;

    DBG << "t" << ++cur_thread << ": " << to_sec(stats.compression_time) << ' '
        << to_sec(stats.sync_time) << ' ' << to_sec(stats.copy_time);
  }

  DBG << "sum: " << to_sec(total_time_compression) << ' ' << to_sec(total_time_sync) << ' '
      << to_sec(total_time_copy);
}

} // namespace debug

std::pair<StaticArray<NodeID>, StaticArray<NodeID>>
sort_by_degree_buckets(const NodeID n, const StaticArray<NodeID> &degrees) {
  auto find_bucket = [&](const NodeID deg) {
    return deg == 0 ? (kNumberOfDegreeBuckets<NodeID> - 1) : degree_bucket(deg);
  };

  const std::size_t cpus = std::min<std::size_t>(tbb::this_task_arena::max_concurrency(), n);
  RECORD("permutation") StaticArray<NodeID> permutation(n, static_array::noinit);
  RECORD("inverse_permutation") StaticArray<NodeID> inverse_permutation(n, static_array::noinit);

  using Buckets = std::array<NodeID, kNumberOfDegreeBuckets<NodeID> + 1>;
  std::vector<Buckets, tbb::cache_aligned_allocator<Buckets>> local_buckets(cpus + 1);

  parallel::deterministic_for<NodeID>(
      0,
      n,
      [&](const NodeID from, const NodeID to, const std::size_t cpu) {
        KASSERT(cpu < cpus);

        for (NodeID u = from; u < to; ++u) {
          const auto bucket = find_bucket(degrees[u]);
          permutation[u] = local_buckets[cpu + 1][bucket]++;
        }
      }
  );

  // Build a table of prefix numbers to correct the position of each node in the
  // final permutation After the previous loop, permutation[u] contains the
  // position of u in the thread-local bucket. (i) account for smaller buckets
  // --> add prefix computed in global_buckets (ii) account for the same bucket
  // in smaller processor IDs --> add prefix computed in local_buckets
  Buckets global_buckets{};
  for (std::size_t id = 1; id < cpus + 1; ++id) {
    for (std::size_t i = 0; i + 1 < global_buckets.size(); ++i) {
      global_buckets[i + 1] += local_buckets[id][i];
    }
  }
  parallel::prefix_sum(global_buckets.begin(), global_buckets.end(), global_buckets.begin());
  for (std::size_t i = 0; i < global_buckets.size(); ++i) {
    for (std::size_t id = 0; id + 1 < cpus; ++id) {
      local_buckets[id + 1][i] += local_buckets[id][i];
    }
  }

  // Apply offsets to obtain global permutation
  parallel::deterministic_for<NodeID>(
      0,
      n,
      [&](const NodeID from, const NodeID to, const std::size_t cpu) {
        KASSERT(cpu < cpus);

        for (NodeID u = from; u < to; ++u) {
          const NodeID bucket = find_bucket(degrees[u]);
          permutation[u] += global_buckets[bucket] + local_buckets[cpu][bucket];
        }
      }
  );

  // Compute inverse permutation
  tbb::parallel_for<NodeID>(0, n, [&](const NodeID u) { inverse_permutation[permutation[u]] = u; });

  return {std::move(permutation), std::move(inverse_permutation)};
}

} // namespace

CompressedGraph compressed_read_parallel(const std::string &filename, const NodeOrdering ordering) {
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
    const auto fetch_degree = [&](const NodeID node) {
      return static_cast<NodeID>((nodes[node + 1] - nodes[node]) / sizeof(NodeID));
    };

    RECORD("degrees") StaticArray<NodeID> degrees(num_nodes, static_array::noinit);
    TIMED_SCOPE("Read degrees") {
      tbb::parallel_for(tbb::blocked_range<NodeID>(0, num_nodes), [&](const auto &r) {
        for (NodeID u = r.begin(); u != r.end(); ++u) {
          degrees[u] = fetch_degree(u);
        }
      });
    };

    const bool sort_by_degree_bucket = ordering == NodeOrdering::DEGREE_BUCKETS;
    StaticArray<NodeID> permutation;
    StaticArray<NodeID> inverse_permutation;
    if (sort_by_degree_bucket) {
      SCOPED_TIMER("Compute permutation");
      auto [perm, inv_perm] = sort_by_degree_buckets(num_nodes, degrees);
      permutation = std::move(perm);
      inverse_permutation = std::move(inv_perm);
    }

    // To compress the graph in parallel the nodes are split into chunks. Each parallel task fetches
    // a chunk and compresses the neighbourhoods of the corresponding nodes. The compressed
    // neighborhoods are meanwhile temporarily stored in a buffer. They are moved into the
    // compressed edge array when the (total) length of the compressed neighborhoods of the previous
    // chunks is determined.
    constexpr std::size_t kNumChunks = 5000;
    const EdgeID max_chunk_size = num_edges / kNumChunks;
    std::vector<std::tuple<NodeID, NodeID, EdgeID>> chunks;

    NodeID max_degree = 0;
    TIMED_SCOPE("Compute chunks") {
      NodeID cur_chunk_start = 0;
      EdgeID cur_chunk_size = 0;
      EdgeID cur_first_edge = 0;
      for (NodeID i = 0; i < num_nodes; ++i) {
        NodeID node = sort_by_degree_bucket ? inverse_permutation[i] : i;

        const NodeID degree = degrees[node];
        max_degree = std::max(max_degree, degree);

        cur_chunk_size += degree;
        if (cur_chunk_size >= max_chunk_size) {
          if (cur_chunk_start == i) {
            chunks.emplace_back(cur_chunk_start, i + 1, cur_first_edge);

            cur_chunk_start = i + 1;
            cur_first_edge += degree;
            cur_chunk_size = 0;
          } else {
            chunks.emplace_back(cur_chunk_start, i, cur_first_edge);

            cur_chunk_start = i;
            cur_first_edge += cur_chunk_size - degree;
            cur_chunk_size = degree;
          }
        }
      }

      if (cur_chunk_start != num_nodes) {
        chunks.emplace_back(cur_chunk_start, num_nodes, cur_first_edge);
      }
    };

    degrees.free();

    // Initializes the data structures used to build the compressed graph in parallel.
    const bool sorted = ordering == NodeOrdering::IMPLICIT_DEGREE_BUCKETS || sort_by_degree_bucket;
    ParallelCompressedGraphBuilder builder(
        header.num_nodes, header.num_edges, header.has_node_weights, header.has_edge_weights, sorted
    );

    tbb::enumerable_thread_specific<std::vector<EdgeID>> offsets_ets;
    tbb::enumerable_thread_specific<std::vector<std::pair<NodeID, EdgeWeight>>> neighbourhood_ets;
    tbb::enumerable_thread_specific<CompressedEdgesBuilder> neighbourhood_builder_ets([&] {
      return CompressedEdgesBuilder(
          num_nodes, num_edges, max_degree, header.has_edge_weights, builder.edge_weights()
      );
    });

    const std::size_t num_threads = tbb::this_task_arena::max_concurrency();
    ConcurrentCircularVectorMutex<NodeID, EdgeID> buffer(num_threads);

    tbb::enumerable_thread_specific<debug::Stats> dbg_ets;
    tbb::parallel_for<NodeID>(0, chunks.size(), [&](const auto) {
      auto &dbg = dbg_ets.local();
      IF_DBG dbg.num_chunks++;

      std::vector<EdgeID> &offsets = offsets_ets.local();
      std::vector<std::pair<NodeID, EdgeWeight>> &neighbourhood = neighbourhood_ets.local();
      CompressedEdgesBuilder &neighbourhood_builder = neighbourhood_builder_ets.local();

      const NodeID chunk = buffer.next();
      const auto [start, end, first_edge] = chunks[chunk];

      NodeWeight local_node_weight = 0;
      neighbourhood_builder.init(first_edge);

      // Compress the neighborhoods of the nodes in the fetched chunk.
      debug::scoped_time(dbg.compression_time, [&] {
        for (NodeID i = start; i < end; ++i) {
          const NodeID node = sort_by_degree_bucket ? inverse_permutation[i] : i;
          const NodeID degree = fetch_degree(node);
          IF_DBG dbg.num_edges += degree;

          EdgeID edge = map_edge_offset(node);
          for (NodeID j = 0; j < degree; ++j) {
            const NodeID adjacent_node =
                sort_by_degree_bucket ? permutation[edges[edge]] : edges[edge];
            const EdgeWeight edge_weight = header.has_edge_weights ? edge_weights[edge] : 1;

            neighbourhood.emplace_back(adjacent_node, edge_weight);
            edge += 1;
          }

          const EdgeID local_offset = neighbourhood_builder.add(i, neighbourhood);
          offsets.push_back(local_offset);

          neighbourhood.clear();
        }
      });

      // Wait for the parallel tasks that process the previous chunks to finish.
      const EdgeID offset = debug::scoped_time(dbg.sync_time, [&] {
        const EdgeID compressed_neighborhoods_size = neighbourhood_builder.size();
        return buffer.fetch_and_update(chunk, compressed_neighborhoods_size);
      });

      // Store the edge offset and node weight for each node in the chunk and copy the compressed
      // neighborhoods into the actual compressed edge array.
      debug::scoped_time(dbg.copy_time, [&] {
        for (NodeID i = start; i < end; ++i) {
          const EdgeID local_offset = offsets[i - start];

          builder.add_node(i, offset + local_offset);

          if (header.has_node_weights) {
            const NodeID node = sort_by_degree_bucket ? inverse_permutation[i] : i;
            const NodeWeight node_weight = node_weights[node];
            local_node_weight += node_weight;

            builder.add_node_weight(i, node_weight);
          }
        }
        offsets.clear();

        builder.add_compressed_edges(
            offset, neighbourhood_builder.size(), neighbourhood_builder.compressed_data()
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
    });

    IF_DBG debug::print_stats(dbg_ets);

    return builder.build();
  } catch (const BinaryReaderException &e) {
    LOG_ERROR << e.what();
    std::exit(1);
  }
}

} // namespace kaminpar::shm::io::parhip
