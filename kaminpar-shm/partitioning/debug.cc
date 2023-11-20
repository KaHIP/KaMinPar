/*******************************************************************************
 * Debug utilities.
 *
 * @file:   debug.cc
 * @author: Daniel Seemaier
 * @date:   18.04.2023
 ******************************************************************************/
#include "kaminpar-shm/partitioning/debug.h"

#include <fstream>
#include <iomanip>
#include <string>

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/graph.h"

#include "kaminpar-common/random.h"
#include "kaminpar-common/strutils.h"

namespace kaminpar::shm::debug {
namespace {
std::string generate_filename(const Graph &graph, const Context &ctx, const std::string &suffix) {
  std::stringstream filename_ss;

  if (!ctx.debug.dump_dir.empty()) {
    filename_ss << ctx.debug.dump_dir << "/";
  }

  if (ctx.debug.graph_name.empty()) {
    filename_ss << "undefined_n" << graph.n() << "_m" << graph.m();
  } else {
    if (ctx.debug.dump_dir.empty()) {
      filename_ss << ctx.debug.graph_name;
    } else {
      // This is currently the same as above, since graph_name is usually already the extracted
      // basename of the graph; we keep the branch none-the-less in case this changes in the future
      filename_ss << str::extract_basename(ctx.debug.graph_name);
    }
  }

  filename_ss << "." << suffix;

  if (ctx.debug.include_num_threads_in_filename) {
    filename_ss << ".P" << ctx.parallel.num_threads;
  }
  if (ctx.debug.include_seed_in_filename) {
    filename_ss << ".seed" << Random::get_seed();
  }
  if (ctx.debug.include_epsilon_in_filename) {
    filename_ss << ".eps" << std::fixed << std::setprecision(3) << ctx.partition.epsilon;
  }
  if (ctx.debug.include_k_in_filename) {
    filename_ss << ".k" << ctx.partition.k;
  }

  return filename_ss.str();
}
} // namespace

void dump_coarsest_graph(const Graph &graph, const Context &ctx) {
  if (ctx.debug.dump_coarsest_graph) {
    dump_graph(graph, generate_filename(graph, ctx, "coarsest.metis"));
  }
}

void dump_graph_hierarchy(const Graph &graph, const int level, const Context &ctx) {
  if (ctx.debug.dump_graph_hierarchy) {
    dump_graph(graph, generate_filename(graph, ctx, "level" + std::to_string(level) + ".metis"));
  }
  if (level == 0 && ctx.debug.dump_toplevel_graph) {
    dump_graph(graph, generate_filename(graph, ctx, "toplevel.metis"));
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

void dump_coarsest_partition(const PartitionedGraph &p_graph, const Context &ctx) {
  if (ctx.debug.dump_coarsest_partition) {
    dump_partition(p_graph, generate_filename(p_graph.graph(), ctx, "coarsest.part"));
  }
}

void dump_partition_hierarchy(
    const PartitionedGraph &p_graph, const int level, const std::string &state, const Context &ctx
) {
  if (ctx.debug.dump_partition_hierarchy) {
    dump_partition(
        p_graph,
        generate_filename(
            p_graph.graph(), ctx, "level" + std::to_string(level) + "." + state + ".part"
        )
    );
  }
  if (level == 0 && ctx.debug.dump_toplevel_partition) {
    dump_partition(p_graph, generate_filename(p_graph.graph(), ctx, "toplevel." + state + ".part"));
  }
}

void dump_partition(const PartitionedGraph &p_graph, const std::string &filename) {
  std::ofstream out(filename, std::ios::trunc);
  for (const NodeID u : p_graph.nodes()) {
    out << p_graph.block(u) << "\n";
  }
}
} // namespace kaminpar::shm::debug
