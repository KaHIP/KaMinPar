/*******************************************************************************
 * Sequential METIS parser.
 *
 * @file:   metis_parser.cc
 * @author: Daniel Seemaier
 * @date:   26.10.2022
 ******************************************************************************/
#include "apps/io/shm_metis_parser.h"

#include <fstream>

#include "kaminpar-shm/graphutils/compressed_graph_builder.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"

#include "apps/io/file_tokener.h"

namespace kaminpar::shm::io::metis {
using namespace kaminpar::io;

namespace {

struct MetisHeader {
  std::uint64_t num_nodes = 0;
  std::uint64_t num_edges = 0;
  bool has_node_weights = false;
  bool has_edge_weights = false;
};

MetisHeader parse_header(MappedFileToker &toker) {
  toker.skip_spaces();
  while (toker.current() == '%') {
    toker.skip_line();
    toker.skip_spaces();
  }

  const std::uint64_t num_nodes = toker.scan_uint();
  const std::uint64_t num_edges = toker.scan_uint();
  const std::uint64_t format = (toker.current() != '\n') ? toker.scan_uint() : 0;
  toker.consume_char('\n');

  if (format != 0 && format != 1 && format != 10 && format != 11 && format && format != 100 &&
      format != 110 && format != 101 && format != 111) {
    LOG_WARNING << "invalid or unsupported graph format";
  }

  [[maybe_unused]] const bool has_node_sizes = format / 100; // == 1xx
  const bool has_node_weights = (format % 100) / 10;         // == x1x
  const bool has_edge_weights = format % 10;                 // == xx1

  if (has_node_sizes) {
    LOG_WARNING << "ignoring node sizes";
  }

  KASSERT(
      num_nodes <= static_cast<std::uint64_t>(std::numeric_limits<NodeID>::max()),
      "number of nodes is too large for the node ID type"
  );
  KASSERT(
      num_edges <= static_cast<std::uint64_t>(std::numeric_limits<EdgeID>::max()),
      "number of edges is too large for the edge ID type"
  );
  KASSERT(
      num_edges <= (num_nodes * (num_nodes - 1)) / 2,
      "specified number of edges is impossibly large"
  );

  return {
      .num_nodes = num_nodes,
      .num_edges = num_edges,
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
    toker.skip_spaces();
    while (toker.current() == '%') {
      toker.skip_line();
      toker.skip_spaces();
    }

    std::uint64_t node_weight = 1;
    if (header.has_node_weights) {
      node_weight = toker.scan_uint();

      KASSERT(
          node_weight <= static_cast<std::uint64_t>(std::numeric_limits<NodeWeight>::max()),
          "node weight is too large for the node weight type"
      );
      KASSERT(node_weight > 0u, "zero node weights are not supported");
    }
    if constexpr (stoppable) {
      if (next_node_cb(node_weight)) {
        has_exited_preemptively = true;
        break;
      }
    } else {
      next_node_cb(node_weight);
    }

    while (std::isdigit(toker.current())) {
      const std::uint64_t v = toker.scan_uint() - 1;

      std::uint64_t edge_weight = 1;
      if (header.has_edge_weights) {
        edge_weight = toker.scan_uint();

        KASSERT(
            edge_weight <= static_cast<std::uint64_t>(std::numeric_limits<EdgeWeight>::max()),
            "edge weight is too large for the edge weight type"
        );
        KASSERT(edge_weight > 0u, "zero edge weights are not supported");
      }

      KASSERT(v < header.num_nodes, "neighbor out of bounds");
      KASSERT(u != v, "detected illegal self-loop");

      next_edge_cb(edge_weight, v);
    }

    if (toker.valid_position()) {
      toker.consume_char('\n');
    }
  }

  if (!has_exited_preemptively) {
    while (toker.current() == '%') {
      toker.skip_line();
    }

    if (toker.valid_position()) {
      LOG_WARNING << "ignorning extra lines in input file";
    }
  }
}

} // namespace

CSRGraph csr_read(const std::string &filename, const bool sorted) {
  MappedFileToker toker(filename);
  const MetisHeader header = parse_header(toker);

  RECORD("nodes") StaticArray<EdgeID> nodes(header.num_nodes + 1, static_array::noinit);
  RECORD("edges") StaticArray<NodeID> edges(header.num_edges * 2, static_array::noinit);

  RECORD("node_weights") StaticArray<NodeWeight> node_weights;
  if (header.has_node_weights) {
    node_weights.resize(header.num_nodes, static_array::noinit);
  }

  RECORD("edge_weights") StaticArray<EdgeWeight> edge_weights;
  if (header.has_edge_weights) {
    edge_weights.resize(header.num_edges * 2, static_array::noinit);
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
          total_node_weight += weight;
          node_weights[u] = static_cast<NodeWeight>(weight);
        }

        u += 1;
      },
      [&](const std::uint64_t weight, const std::uint64_t v) {
        edges[e] = static_cast<NodeID>(v);

        if (header.has_edge_weights) {
          total_edge_weight += weight;
          edge_weights[e] = static_cast<EdgeWeight>(weight);
        }

        e += 1;
      }
  );

  KASSERT(u + 1 == nodes.size());
  KASSERT(e == header.num_edges * 2);
  nodes[u] = e;

  // Only keep weights if the graph is really weighted.
  const bool unit_node_weights =
      header.has_node_weights && (static_cast<NodeID>(total_node_weight + 1) == nodes.size());
  if (unit_node_weights) {
    node_weights.free();
  }

  const bool unit_edge_weights =
      header.has_edge_weights && (static_cast<EdgeID>(total_edge_weight) == edges.size());
  if (unit_edge_weights) {
    edge_weights.free();
  }

  KASSERT(
      total_node_weight <= static_cast<std::int64_t>(std::numeric_limits<NodeWeight>::max()),
      "total node weight does not fit into the node weight type"
  );
  KASSERT(
      total_edge_weight <= static_cast<std::int64_t>(std::numeric_limits<EdgeWeight>::max()),
      "total edge weight does not fit into the edge weight type"
  );

  return CSRGraph(
      std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), sorted
  );
}

CompressedGraph compress_read(const std::string &filename, const bool sorted) {
  MappedFileToker toker(filename);
  const MetisHeader header = parse_header(toker);

  const std::size_t uncompressed_graph_size =
      (header.num_nodes + 1) * sizeof(EdgeID) + header.num_edges * 2 * sizeof(NodeID) +
      header.has_node_weights * header.num_nodes * sizeof(NodeWeight) +
      header.has_edge_weights * header.num_edges * 2 * sizeof(EdgeID);
  bool dismissed = false;

  CompressedGraphBuilder builder(
      header.num_nodes,
      header.num_edges * 2,
      header.has_node_weights,
      header.has_edge_weights,
      sorted
  );
  RECORD("neighbourhood") std::vector<std::pair<NodeID, EdgeWeight>> neighbourhood;
  RECORD_LOCAL_DATA_STRUCT("vector<pair<NodeID, EdgeWeight>>", 0, neighbourhood_stats);

  NodeID node = 0;
  EdgeID edge = 0;
  parse_graph(
      toker,
      header,
      [&](const std::uint64_t weight) {
        if (node > 0) {
          builder.add_node(node - 1, neighbourhood);
          neighbourhood.clear();
        }

        if (header.has_node_weights) {
          builder.add_node_weight(node, static_cast<NodeWeight>(weight));
        }

        node += 1;
        return false;
      },
      [&](const std::uint64_t weight, const std::uint64_t v) {
        neighbourhood.emplace_back(static_cast<NodeID>(v), static_cast<EdgeWeight>(weight));
        edge += 1;
      }
  );
  builder.add_node(node - 1, neighbourhood);

  KASSERT(
      builder.total_node_weight() <=
          static_cast<std::int64_t>(std::numeric_limits<NodeWeight>::max()),
      "total node weight does not fit into the node weight type"
  );
  KASSERT(
      builder.total_edge_weight() <=
          static_cast<std::int64_t>(std::numeric_limits<EdgeWeight>::max()),
      "total edge weight does not fit into the edge weight type"
  );
  IF_HEAP_PROFILING(neighbourhood_stats->size = neighbourhood.capacity() * sizeof(NodeID));

  return builder.build();
}

void write(const std::string &filename, const Graph &graph) {
  std::ofstream out(filename);

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
}

} // namespace kaminpar::shm::io::metis
