/*******************************************************************************
 * Sequential METIS parser.
 *
 * @file:   metis_parser.cc
 * @author: Daniel Seemaier
 * @date:   26.10.2022
 ******************************************************************************/
#include "kaminpar-io/metis_parser.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

#include "kaminpar-io/util/file_toker.h"
#include "kaminpar-io/util/io_validation.h"

#include "kaminpar-shm/datastructures/compressed_graph.h"
#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/graphutils/compressed_graph_builder.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::shm::io::metis {
using namespace kaminpar::io;

namespace {

struct MetisHeader {
  std::uint64_t num_nodes = 0;
  std::uint64_t num_edges = 0;
  bool has_node_sizes = false;
  bool has_node_weights = false;
  bool has_edge_weights = false;
};

template <typename Weight>
void add_weight(std::int64_t &total, const std::uint64_t weight, const char *message) {
  const auto max = static_cast<std::uint64_t>(std::numeric_limits<Weight>::max());
  raise_if(weight > max, message);
  raise_if(static_cast<std::uint64_t>(total) > max - weight, message);
  total += static_cast<std::int64_t>(weight);
}

std::uint64_t checked_adjacency_entries(const MetisHeader header) {
  raise_if(
      header.num_edges > std::numeric_limits<std::uint64_t>::max() / 2,
      "the number of adjacency entries is too large"
  );
  return header.num_edges * 2;
}

bool is_valid_metis_format(const std::uint64_t format) {
  return format == 0 || format == 1 || format == 10 || format == 11 || format == 100 ||
         format == 101 || format == 110 || format == 111;
}

bool is_line_end(const MappedFileToker &toker) {
  return !toker.valid_position() || toker.current() == '\n' || toker.current() == '\r';
}

void consume_line_end(MappedFileToker &toker) {
  if (!toker.valid_position()) {
    return;
  }
  if (toker.current() == '\r') {
    toker.advance();
  }
  if (toker.valid_position()) {
    toker.consume_char('\n');
  }
}

void skip_comment_and_empty_lines(MappedFileToker &toker) {
  while (toker.valid_position()) {
    toker.skip_spaces();
    if (!toker.valid_position()) {
      return;
    }
    if (toker.current() == '%' || toker.current() == '#') {
      toker.skip_line();
    } else if (is_line_end(toker)) {
      consume_line_end(toker);
    } else {
      return;
    }
  }
}

void skip_comment_lines(MappedFileToker &toker) {
  while (toker.valid_position()) {
    toker.skip_spaces();
    if (toker.valid_position() && (toker.current() == '%' || toker.current() == '#')) {
      toker.skip_line();
    } else {
      return;
    }
  }
}

void finish_line(MappedFileToker &toker) {
  toker.skip_spaces();
  raise_if(!is_line_end(toker), "Unexpected character in METIS graph file");
  consume_line_end(toker);
}

MetisHeader parse_header(MappedFileToker &toker) {
  skip_comment_and_empty_lines(toker);

  const std::uint64_t num_nodes = toker.scan_uint();
  const std::uint64_t num_edges = toker.scan_uint();
  const std::uint64_t format = !is_line_end(toker) ? toker.scan_uint() : 0;
  finish_line(toker);

  raise_if(!is_valid_metis_format(format), "Invalid or unsupported METIS graph format");

  const bool has_node_sizes = format / 100;          // == 1xx
  const bool has_node_weights = (format % 100) / 10; // == x1x
  const bool has_edge_weights = format % 10;         // == xx1

  if (has_node_sizes) {
    LOG_WARNING << "ignoring node sizes";
  }

  ensure_fits<NodeID>(num_nodes, "number of nodes is too large for the node ID type");
  raise_if(num_nodes == std::numeric_limits<std::uint64_t>::max(), "number of nodes is too large");
  ensure_fits<EdgeID>(num_edges, "number of edges is too large for the edge ID type");
  raise_if(
      static_cast<unsigned __int128>(num_edges) >
          (static_cast<unsigned __int128>(num_nodes) * (num_nodes - 1)) / 2,
      "specified number of edges is impossibly large"
  );
  ensure_fits<EdgeID>(
      checked_adjacency_entries({.num_nodes = num_nodes, .num_edges = num_edges}),
      "the number of adjacency entries is too large for the edge ID type"
  );

  return {
      .num_nodes = num_nodes,
      .num_edges = num_edges,
      .has_node_sizes = has_node_sizes,
      .has_node_weights = has_node_weights,
      .has_edge_weights = has_edge_weights,
  };
}

template <typename NextNodeCB, typename NextEdgeCB>
void parse_graph(
    MappedFileToker &toker,
    const MetisHeader header,
    NextNodeCB &&next_node_cb,
    NextEdgeCB &&next_edge_cb
) {
  static_assert(std::is_invocable_v<NextNodeCB, std::uint64_t>);
  static_assert(std::is_invocable_v<NextEdgeCB, std::uint64_t, std::uint64_t>);
  constexpr bool stoppable = std::is_invocable_r_v<bool, NextNodeCB, std::uint64_t>;

  bool has_exited_preemptively = false;
  for (std::uint64_t u = 0; u < header.num_nodes; ++u) {
    skip_comment_lines(toker);
    raise_if(
        !toker.valid_position(), "input contains fewer vertex lines than specified in the header"
    );

    if (header.has_node_sizes) {
      toker.skip_uint();
    }

    std::uint64_t node_weight = 1;
    if (header.has_node_weights) {
      node_weight = toker.scan_uint();

      ensure_fits<NodeWeight>(node_weight, "node weight is too large for the node weight type");
      raise_if(node_weight == 0u, "zero node weights are not supported");
    }
    if constexpr (stoppable) {
      if (next_node_cb(node_weight)) {
        has_exited_preemptively = true;
        break;
      }
    } else {
      next_node_cb(node_weight);
    }

    while (toker.current_is_digit()) {
      const std::uint64_t raw_v = toker.scan_uint();
      raise_if(raw_v == 0, "METIS vertex IDs must be one-based");
      const std::uint64_t v = raw_v - 1;

      std::uint64_t edge_weight = 1;
      if (header.has_edge_weights) {
        edge_weight = toker.scan_uint();

        ensure_fits<EdgeWeight>(edge_weight, "edge weight is too large for the edge weight type");
        raise_if(edge_weight == 0u, "zero edge weights are not supported");
      }

      raise_if(v >= header.num_nodes, "neighbor out of bounds");
      raise_if(u == v, "detected illegal self-loop");

      next_edge_cb(edge_weight, v);
    }

    finish_line(toker);
  }

  if (!has_exited_preemptively) {
    skip_comment_and_empty_lines(toker);
    raise_if(
        toker.valid_position(), "input contains more vertex lines than specified in the header"
    );
  }
}

struct FileChunk {
  std::size_t begin;
  std::size_t end;
};

struct ParsedChunk {
  std::vector<EdgeID> nodes;
  std::vector<NodeID> edges;
  std::vector<NodeWeight> node_weights;
  std::vector<EdgeWeight> edge_weights;
  std::int64_t total_node_weight = 0;
  std::int64_t total_edge_weight = 0;

  [[nodiscard]] NodeID num_nodes() const {
    return static_cast<NodeID>(nodes.size() - 1);
  }

  [[nodiscard]] EdgeID num_edges() const {
    return static_cast<EdgeID>(edges.size());
  }
};

struct ChunkToFinalize {
  ParsedChunk parsed;
  NodeID node_offset = 0;
  EdgeID edge_offset = 0;
};

std::vector<FileChunk>
compute_file_chunks(const MappedFileToker &toker, const std::size_t data_begin) {
  const std::size_t data_end = toker.length();
  if (data_begin >= data_end) {
    return {};
  }

  const std::size_t data_size = data_end - data_begin;
  const MappedFileToker data_toker(toker, data_begin, data_end);
  const std::size_t num_threads = std::max<std::size_t>(1, tbb::this_task_arena::max_concurrency());
  constexpr std::size_t kMinChunkSize = 1 << 20;
  const std::size_t num_chunks = std::max<std::size_t>(
      1, std::min<std::size_t>(num_threads * 4, (data_size + kMinChunkSize - 1) / kMinChunkSize)
  );

  std::vector<FileChunk> chunks;
  chunks.reserve(num_chunks);

  std::size_t chunk_begin = data_begin;
  for (std::size_t i = 1; i <= num_chunks; ++i) {
    const std::size_t raw_end = data_begin + (i * data_size) / num_chunks;
    const std::size_t chunk_end =
        i == num_chunks ? data_end : data_toker.advance_to_line_begin(raw_end);

    if (chunk_begin < chunk_end) {
      chunks.push_back({.begin = chunk_begin, .end = chunk_end});
    }

    chunk_begin = chunk_end;
  }

  return chunks;
}

ParsedChunk
parse_chunk(const MappedFileToker &mapped_file, const FileChunk chunk, const MetisHeader header) {
  MappedFileToker toker(mapped_file, chunk.begin, chunk.end);

  ParsedChunk parsed;
  parsed.nodes.push_back(0);

  while (toker.valid_position()) {
    skip_comment_lines(toker);

    if (!toker.valid_position()) {
      break;
    }

    if (header.has_node_sizes) {
      toker.skip_uint();
    }

    std::uint64_t node_weight = 1;
    if (header.has_node_weights) {
      node_weight = toker.scan_uint();

      ensure_fits<NodeWeight>(node_weight, "node weight is too large for the node weight type");
      raise_if(node_weight == 0u, "zero node weights are not supported");

      parsed.node_weights.push_back(static_cast<NodeWeight>(node_weight));
      add_weight<NodeWeight>(
          parsed.total_node_weight,
          node_weight,
          "total node weight does not fit into the node weight type"
      );
    }

    while (toker.current_is_digit()) {
      const std::uint64_t raw_v = toker.scan_uint();
      raise_if(raw_v == 0, "METIS vertex IDs must be one-based");
      const std::uint64_t v = raw_v - 1;

      std::uint64_t edge_weight = 1;
      if (header.has_edge_weights) {
        edge_weight = toker.scan_uint();

        ensure_fits<EdgeWeight>(edge_weight, "edge weight is too large for the edge weight type");
        raise_if(edge_weight == 0u, "zero edge weights are not supported");

        parsed.edge_weights.push_back(static_cast<EdgeWeight>(edge_weight));
        add_weight<EdgeWeight>(
            parsed.total_edge_weight,
            edge_weight,
            "total edge weight does not fit into the edge weight type"
        );
      }

      raise_if(v >= header.num_nodes, "neighbor out of bounds");
      parsed.edges.push_back(static_cast<NodeID>(v));
    }

    raise_if(
        parsed.edges.size() > static_cast<std::uint64_t>(std::numeric_limits<EdgeID>::max()),
        "too many adjacency entries in chunk"
    );
    parsed.nodes.push_back(static_cast<EdgeID>(parsed.edges.size()));

    finish_line(toker);
  }

  return parsed;
}

void finalize_chunk(
    ParsedChunk &parsed,
    const NodeID node_offset,
    const EdgeID edge_offset,
    const MetisHeader header,
    StaticArray<EdgeID> &nodes,
    StaticArray<NodeID> &edges,
    StaticArray<NodeWeight> &node_weights,
    StaticArray<EdgeWeight> &edge_weights
) {
  for (NodeID i = 0; i < parsed.num_nodes(); ++i) {
    nodes[node_offset + i] = edge_offset + parsed.nodes[i];

    for (EdgeID e = parsed.nodes[i]; e < parsed.nodes[i + 1]; ++e) {
      raise_if(node_offset + i == parsed.edges[e], "detected illegal self-loop");
    }
  }

  std::copy(parsed.edges.begin(), parsed.edges.end(), edges.begin() + edge_offset);

  if (header.has_node_weights) {
    std::copy(
        parsed.node_weights.begin(), parsed.node_weights.end(), node_weights.begin() + node_offset
    );
  }

  if (header.has_edge_weights) {
    std::copy(
        parsed.edge_weights.begin(), parsed.edge_weights.end(), edge_weights.begin() + edge_offset
    );
  }
}

} // namespace

Graph csr_read(const std::string &filename, const bool sorted) {
  MappedFileToker toker(filename);
  const MetisHeader header = parse_header(toker);
  const std::uint64_t expected_nnz = checked_adjacency_entries(header);

  StaticArray<EdgeID> nodes(header.num_nodes + 1, static_array::noinit);
  StaticArray<NodeID> edges(expected_nnz, static_array::noinit);

  StaticArray<NodeWeight> node_weights;
  if (header.has_node_weights) {
    node_weights.resize(header.num_nodes, static_array::noinit);
  }

  StaticArray<EdgeWeight> edge_weights;
  if (header.has_edge_weights) {
    edge_weights.resize(expected_nnz, static_array::noinit);
  }

  NodeID u = 0;
  EdgeID e = 0;

  std::int64_t total_node_weight = 0;
  std::int64_t total_edge_weight = 0;

  parse_graph(
      toker,
      header,
      [&](const std::uint64_t weight) {
        nodes[u] = e;

        if (header.has_node_weights) {
          add_weight<NodeWeight>(
              total_node_weight, weight, "total node weight does not fit into the node weight type"
          );
          node_weights[u] = static_cast<NodeWeight>(weight);
        }

        u += 1;
      },
      [&](const std::uint64_t weight, const std::uint64_t v) {
        raise_if(
            e >= expected_nnz, "input contains more adjacency entries than specified in the header"
        );
        edges[e] = static_cast<NodeID>(v);

        if (header.has_edge_weights) {
          add_weight<EdgeWeight>(
              total_edge_weight, weight, "total edge weight does not fit into the edge weight type"
          );
          edge_weights[e] = static_cast<EdgeWeight>(weight);
        }

        e += 1;
      }
  );

  raise_if(u != header.num_nodes, "input contains fewer vertex lines than specified in the header");
  raise_if(
      e != expected_nnz, "input contains fewer adjacency entries than specified in the header"
  );
  nodes[u] = e;

  if (!header.has_node_weights) {
    ensure_fits<NodeWeight>(
        header.num_nodes, "total node weight does not fit into the node weight type"
    );
  }
  if (!header.has_edge_weights) {
    ensure_fits<EdgeWeight>(
        expected_nnz, "total edge weight does not fit into the edge weight type"
    );
  }

  // Only keep weights if the graph is really weighted.
  const bool unit_node_weights =
      header.has_node_weights &&
      (static_cast<std::uint64_t>(total_node_weight) == header.num_nodes);
  if (unit_node_weights) {
    node_weights.free();
  }

  const bool unit_edge_weights =
      header.has_edge_weights && (static_cast<std::uint64_t>(total_edge_weight) == expected_nnz);
  if (unit_edge_weights) {
    edge_weights.free();
  }

  return Graph(
      std::make_unique<CSRGraph>(
          std::move(nodes),
          std::move(edges),
          std::move(node_weights),
          std::move(edge_weights),
          sorted
      )
  );
}

Graph csr_read_parallel(const std::string &filename, const bool sorted) {
  MappedFileToker toker(filename);
  const MetisHeader header = parse_header(toker);
  const std::size_t data_begin = toker.position();

  const std::uint64_t expected_nnz = checked_adjacency_entries(header);

  StaticArray<EdgeID> nodes(header.num_nodes + 1, static_array::noinit);
  StaticArray<NodeID> edges(expected_nnz, static_array::noinit);

  StaticArray<NodeWeight> node_weights;
  if (header.has_node_weights) {
    node_weights.resize(header.num_nodes, static_array::noinit);
  }

  StaticArray<EdgeWeight> edge_weights;
  if (header.has_edge_weights) {
    edge_weights.resize(expected_nnz, static_array::noinit);
  }

  const std::vector<FileChunk> file_chunks = compute_file_chunks(toker, data_begin);
  std::vector<std::unique_ptr<ParsedChunk>> parsed_chunks(file_chunks.size());

  std::uint64_t next_node = 0;
  std::uint64_t next_edge = 0;

  const std::size_t num_workers = std::min<std::size_t>(
      std::max<std::size_t>(1, tbb::this_task_arena::max_concurrency()),
      std::max<std::size_t>(1, file_chunks.size())
  );
  const std::size_t parse_ahead_window =
      std::max<std::size_t>(1, std::min<std::size_t>(file_chunks.size(), num_workers * 2));

  std::atomic<std::size_t> next_chunk_to_claim{0};
  std::atomic<std::size_t> committed_frontier{0};

  std::size_t next_chunk_to_commit = 0;
  std::int64_t total_node_weight = 0;
  std::int64_t total_edge_weight = 0;

  std::mutex commit_mutex;

  const auto claim_chunk = [&]() -> std::optional<std::size_t> {
    for (;;) {
      const std::size_t chunk_id = next_chunk_to_claim.load(std::memory_order_acquire);
      if (chunk_id >= file_chunks.size()) {
        return std::nullopt;
      }

      const std::size_t frontier = committed_frontier.load(std::memory_order_acquire);
      if (chunk_id >= frontier + parse_ahead_window) {
        std::this_thread::yield();
        continue;
      }

      std::size_t expected = chunk_id;
      if (next_chunk_to_claim.compare_exchange_weak(
              expected, chunk_id + 1, std::memory_order_acq_rel
          )) {
        return chunk_id;
      }
    }
  };

  const auto commit_ready_chunks = [&]() {
    std::vector<ChunkToFinalize> chunks_to_finalize;
    {
      const std::lock_guard<std::mutex> lock(commit_mutex);

      while (next_chunk_to_commit < parsed_chunks.size() &&
             parsed_chunks[next_chunk_to_commit] != nullptr) {
        ParsedChunk parsed = std::move(*parsed_chunks[next_chunk_to_commit]);
        parsed_chunks[next_chunk_to_commit].reset();

        const std::uint64_t local_num_nodes = parsed.num_nodes();
        const std::uint64_t local_num_edges = parsed.num_edges();
        raise_if(
            next_node > header.num_nodes || local_num_nodes > header.num_nodes - next_node,
            "input contains more vertex lines than specified in the header"
        );
        raise_if(
            next_edge > expected_nnz || local_num_edges > expected_nnz - next_edge,
            "input contains more adjacency entries than specified in the header"
        );

        chunks_to_finalize.push_back(
            ChunkToFinalize{
                .parsed = std::move(parsed),
                .node_offset = static_cast<NodeID>(next_node),
                .edge_offset = static_cast<EdgeID>(next_edge),
            }
        );

        next_node += local_num_nodes;
        next_edge += local_num_edges;
        add_weight<NodeWeight>(
            total_node_weight,
            static_cast<std::uint64_t>(chunks_to_finalize.back().parsed.total_node_weight),
            "total node weight does not fit into the node weight type"
        );
        add_weight<EdgeWeight>(
            total_edge_weight,
            static_cast<std::uint64_t>(chunks_to_finalize.back().parsed.total_edge_weight),
            "total edge weight does not fit into the edge weight type"
        );

        ++next_chunk_to_commit;
        committed_frontier.store(next_chunk_to_commit, std::memory_order_release);
      }
    }

    for (ChunkToFinalize &chunk : chunks_to_finalize) {
      finalize_chunk(
          chunk.parsed,
          chunk.node_offset,
          chunk.edge_offset,
          header,
          nodes,
          edges,
          node_weights,
          edge_weights
      );
    }
  };

  tbb::parallel_for<std::size_t>(0, num_workers, [&](const std::size_t) {
    while (const std::optional<std::size_t> chunk_id = claim_chunk()) {
      ParsedChunk parsed = parse_chunk(toker, file_chunks[*chunk_id], header);

      {
        const std::lock_guard<std::mutex> lock(commit_mutex);
        parsed_chunks[*chunk_id] = std::make_unique<ParsedChunk>(std::move(parsed));
      }
      commit_ready_chunks();
    }
  });

  commit_ready_chunks();

  raise_if(
      next_node != header.num_nodes,
      "input contains fewer vertex lines than specified in the header"
  );
  raise_if(
      next_edge != expected_nnz,
      "input contains fewer adjacency entries than specified in the header"
  );
  nodes[static_cast<NodeID>(header.num_nodes)] = static_cast<EdgeID>(expected_nnz);

  if (!header.has_node_weights) {
    ensure_fits<NodeWeight>(
        header.num_nodes, "total node weight does not fit into the node weight type"
    );
  }
  if (!header.has_edge_weights) {
    ensure_fits<EdgeWeight>(
        expected_nnz, "total edge weight does not fit into the edge weight type"
    );
  }

  // Only keep weights if the graph is really weighted.
  const bool unit_node_weights =
      header.has_node_weights &&
      (static_cast<std::uint64_t>(total_node_weight) == header.num_nodes);
  if (unit_node_weights) {
    node_weights.free();
  }

  const bool unit_edge_weights =
      header.has_edge_weights && (static_cast<std::uint64_t>(total_edge_weight) == expected_nnz);
  if (unit_edge_weights) {
    edge_weights.free();
  }

  CSRGraph csr_graph(
      std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), sorted
  );

  return Graph(std::make_unique<CSRGraph>(std::move(csr_graph)));
}

Graph compress_read(const std::string &filename, const bool sorted) {
  MappedFileToker toker(filename);
  const MetisHeader header = parse_header(toker);
  const std::uint64_t expected_nnz = checked_adjacency_entries(header);

  CompressedGraphBuilder builder(
      header.num_nodes, expected_nnz, header.has_node_weights, header.has_edge_weights, sorted
  );
  std::vector<std::pair<NodeID, EdgeWeight>> neighbourhood;

  NodeID node = 0;
  EdgeID edge = 0;
  parse_graph(
      toker,
      header,
      [&](const std::uint64_t weight) {
        if (node > 0) {
          builder.add_node(neighbourhood);
          neighbourhood.clear();
        }

        if (header.has_node_weights) {
          builder.add_node_weight(node, static_cast<NodeWeight>(weight));
        }

        node += 1;
        return false;
      },
      [&](const std::uint64_t weight, const std::uint64_t v) {
        raise_if(
            edge >= expected_nnz,
            "input contains more adjacency entries than specified in the header"
        );
        neighbourhood.emplace_back(static_cast<NodeID>(v), static_cast<EdgeWeight>(weight));
        edge += 1;
      }
  );
  raise_if(
      node != header.num_nodes, "input contains fewer vertex lines than specified in the header"
  );
  raise_if(
      edge != expected_nnz, "input contains fewer adjacency entries than specified in the header"
  );
  if (!header.has_node_weights) {
    ensure_fits<NodeWeight>(
        header.num_nodes, "total node weight does not fit into the node weight type"
    );
  }
  if (!header.has_edge_weights) {
    ensure_fits<EdgeWeight>(
        expected_nnz, "total edge weight does not fit into the edge weight type"
    );
  }
  if (node > 0) {
    builder.add_node(neighbourhood);
  }

  return builder.build();
}

std::optional<Graph> read_graph(
    const std::string &filename,
    const bool compress,
    const NodeOrdering ordering,
    const bool parallel
) {
  try {
    const bool sorted = ordering == NodeOrdering::IMPLICIT_DEGREE_BUCKETS;
    if (compress && parallel) {
      LOG_WARNING << "On-the-fly parallel compression is not implement for METIS graphs. This will "
                     "use extra memory.";

      Graph graph = csr_read_parallel(filename, sorted);
      return parallel_compress(graph.csr_graph());
    } else if (compress) {
      return compress_read(filename, sorted);
    } else if (parallel) {
      return csr_read_parallel(filename, sorted);
    } else {
      return csr_read(filename, sorted);
    }
  } catch (const IOException &) {
    return std::nullopt;
  }
}

void write_graph(const std::string &filename, const Graph &graph) {
  std::ofstream out(filename);

  reified(graph, [&](const auto &graph) {
    out << graph.n() << ' ' << (graph.m() / 2);
    if (graph.is_node_weighted() || graph.is_edge_weighted()) {
      out << ' ';

      if (graph.is_node_weighted()) {
        out << '1';
      }

      out << (graph.is_edge_weighted() ? '1' : '0');
    }
    out << '\n';

    for (const NodeID node : graph.nodes()) {
      if (graph.is_node_weighted()) {
        out << graph.node_weight(node) << ' ';
      }

      graph.adjacent_nodes(node, [&](const NodeID adjacent_node, const EdgeWeight weight) {
        out << (adjacent_node + 1) << ' ';

        if (graph.is_edge_weighted()) {
          out << weight << ' ';
        }
      });

      out << '\n';
    }
  });
}

} // namespace kaminpar::shm::io::metis
