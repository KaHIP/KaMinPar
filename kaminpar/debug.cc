/*******************************************************************************
 * @file:   debug.cc
 * @author: Daniel Seemaier
 * @date:   18.04.2023
 * @brief:  Debug utilities.
 ******************************************************************************/
#include "kaminpar/debug.h"

#include <fstream>
#include <string>

#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"

namespace kaminpar::shm::debug {
namespace {
std::string generate_filename(
    const Graph &graph, const DebugContext &d_ctx, const std::string &suffix
) {
  std::stringstream filename_ss;

  if (d_ctx.graph_name.empty()) {
    filename_ss << "undefined_n" << graph.n() << "_m" << graph.m();
  } else {
    filename_ss << d_ctx.graph_name;
  }
  filename_ss << "." << suffix;
  return filename_ss.str();
}
} // namespace

void dump_coarsest_graph(const Graph &graph, const DebugContext &d_ctx) {
  if (d_ctx.dump_coarsest_graph) {
    dump_graph(graph, generate_filename(graph, d_ctx, "coarsest.metis"));
  }
}

void dump_graph_hierarchy(
    const Graph &graph, const int level, const DebugContext &d_ctx
) {
  if (d_ctx.dump_graph_hierarchy) {
    dump_graph(
        graph,
        generate_filename(
            graph, d_ctx, "level" + std::to_string(level) + ".metis"
        )
    );
  }
}

void dump_graph(const Graph &graph, const std::string &filename) {
  std::ofstream out(filename, std::ios::trunc);
  out << graph.n() << " " << graph.m() / 2 << " ";
  if (graph.is_node_weighted()) {
    out << graph.is_node_weighted() << graph.is_edge_weighted();
  } else if (graph.is_edge_weighted()) {
    out << graph.is_edge_weighted();
  }
  out << "\n";

  for (const NodeID u : graph.nodes()) {
    if (graph.is_node_weighted()) {
      out << graph.node_weight(u) << " ";
    }
    for (const auto &[e, v] : graph.neighbors(u)) {
      out << v + 1 << " ";
      if (graph.is_edge_weighted()) {
        out << graph.edge_weight(e) << " ";
      }
    }
    out << "\n";
  }
}

void dump_coarsest_partition(
    const PartitionedGraph &p_graph, const DebugContext &d_ctx
) {
  if (d_ctx.dump_coarsest_partition) {
    dump_partition(
        p_graph, generate_filename(p_graph.graph(), d_ctx, "coarsest.part")
    );
  }
}

void dump_partition_hierarchy(
    const PartitionedGraph &p_graph,
    const int level,
    const std::string &state,
    const DebugContext &d_ctx
) {
  if (d_ctx.dump_partition_hierarchy) {
    dump_partition(
        p_graph,
        generate_filename(
            p_graph.graph(),
            d_ctx,
            "level" + std::to_string(level) + "." + state + ".part"
        )
    );
  }
}

void dump_partition(
    const PartitionedGraph &p_graph, const std::string &filename
) {
  std::ofstream out(filename, std::ios::trunc);
  for (const NodeID u : p_graph.nodes()) {
    out << p_graph.block(u) << "\n";
  }
}
} // namespace kaminpar::shm::debug

