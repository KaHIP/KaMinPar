/*******************************************************************************
 * Debug utilities.
 *
 * @file:   debug.cc
 * @author: Daniel Seemaier
 * @date:   18.04.2023
 ******************************************************************************/
#include "kaminpar-shm/partitioning/debug.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/random.h"
#include "kaminpar-common/strutils.h"

namespace kaminpar::shm::debug {

namespace {

std::string generate_filename(const std::string &pattern, const Graph &graph, const Context &ctx) {
  std::string filename = pattern;
  return str::replace_all(
      filename,
      {
          {"%graph", (ctx.debug.graph_name.empty() ? "" : ctx.debug.graph_name)},
          {"%n", std::to_string(graph.n())},
          {"%m", std::to_string(graph.m())},
          {"%k", std::to_string(ctx.partition.k)},
          {"%epsilon", std::to_string(ctx.partition.inferred_epsilon())},
          {"%P", std::to_string(ctx.parallel.num_threads)},
          {"%seed", std::to_string(Random::get_seed())},
      }
  );
}

std::string
generate_graph_filename(const std::string &suffix, const Graph &graph, const Context &ctx) {
  return generate_filename(ctx.debug.dump_graph_filename + "." + suffix, graph, ctx);
}

std::string
generate_partition_filename(const std::string &suffix, const Graph &graph, const Context &ctx) {
  return generate_filename(ctx.debug.dump_partition_filename + "." + suffix, graph, ctx);
}

} // namespace

void dump_coarsest_graph(const Graph &graph, const Context &ctx) {
  if (ctx.debug.dump_coarsest_graph) {
    dump_graph(graph, generate_graph_filename("coarsest.metis", graph, ctx));
  }
}

void dump_graph_hierarchy(const Graph &graph, const int level, const Context &ctx) {
  if (ctx.debug.dump_graph_hierarchy) {
    dump_graph(
        graph, generate_graph_filename("level" + std::to_string(level) + ".metis", graph, ctx)
    );
  }
  if (level == 0 && ctx.debug.dump_toplevel_graph) {
    dump_graph(graph, generate_graph_filename("toplevel.metis", graph, ctx));
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

    graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      out << v + 1 << " ";
      if (graph.is_edge_weighted()) {
        out << w << " ";
      }
    });

    out << "\n";
  }
}

void dump_coarsest_partition(const PartitionedGraph &p_graph, const Context &ctx) {
  if (ctx.debug.dump_coarsest_partition) {
    dump_partition(p_graph, generate_partition_filename("coarsest.graph", p_graph.graph(), ctx));
  }
}

void dump_partition_hierarchy(
    const PartitionedGraph &p_graph, const int level, const std::string &state, const Context &ctx
) {
  if (ctx.debug.dump_partition_hierarchy) {
    dump_partition(
        p_graph,
        generate_partition_filename(
            "level" + std::to_string(level) + "." + state + ".part", p_graph.graph(), ctx
        )
    );
  }
  if (level == 0 && ctx.debug.dump_toplevel_partition) {
    dump_partition(
        p_graph, generate_partition_filename("toplevel." + state + ".part", p_graph.graph(), ctx)
    );
  }
}

void dump_partition(const PartitionedGraph &p_graph, const std::string &filename) {
  std::ofstream out(filename, std::ios::trunc);
  for (const NodeID u : p_graph.nodes()) {
    out << p_graph.block(u) << "\n";
  }
}

std::string describe_partition_context(const PartitionContext &p_ctx) {
  std::stringstream ss;

  ss << p_ctx.k << "-way context (inferred epsilon = " << p_ctx.inferred_epsilon() << "):\n";
  ss << "  Total node weight: " << p_ctx.total_node_weight << " (ctx)\n";
  ss << "  Number of nodes:   " << p_ctx.n << " (ctx)\n";
  ss << "  Number of edges:   " << p_ctx.m << " (ctx)\n";
  ss << "  Max block weights: [";
  for (BlockID block = 0; block < p_ctx.k; ++block) {
    ss << p_ctx.max_block_weight(block) << ", ";
  }
  ss << "\b\b]\n";
  ss << "  PB block weights:  [";
  for (BlockID block = 0; block < p_ctx.k; ++block) {
    ss << p_ctx.perfectly_balanced_block_weight(block) << ", ";
  }
  ss << "\b\b]\n";
  return ss.str();
}

std::string
describe_partition_state(const PartitionedGraph &p_graph, const PartitionContext &p_ctx) {
  std::stringstream ss;

  ss << p_graph.k() << "-way partition with " << p_ctx.k
     << "-way context (inferred epsilon = " << p_ctx.inferred_epsilon() << "):\n";
  ss << "  Total node weight: " << p_graph.total_node_weight() << " (graph) <-> "
     << p_ctx.total_node_weight << " (ctx)\n";
  ss << "  Number of nodes:   " << p_graph.n() << " (graph) <-> " << p_ctx.n << " (ctx)\n";
  ss << "  Number of edges:   " << p_graph.m() << " (graph) <-> " << p_ctx.m << " (ctx)\n";
  if (p_graph.k() == p_ctx.k) {
    ss << "  Block weights:     [";
    for (BlockID block : p_graph.blocks()) {
      ss << p_graph.block_weight(block);
      if (p_graph.block_weight(block) < p_ctx.max_block_weight(block)) {
        ss << " < ";
      } else if (p_graph.block_weight(block) > p_ctx.max_block_weight(block)) {
        ss << " > ";
      } else {
        ss << " = ";
      }
      ss << p_ctx.max_block_weight(block) << ", ";
    }
    ss << "\b\b]\n";
  } else {
    ss << "  Block weights:     [";
    for (BlockID block : p_graph.blocks()) {
      ss << p_graph.block_weight(block) << ", ";
    }
    ss << "\b\b]\n";
    ss << "  Max block weights: [";
    for (BlockID block : p_graph.blocks()) {
      ss << p_ctx.max_block_weight(block) << ", ";
    }
    ss << "\b\b]\n";
  }
  ss << "  PB block weights:  [";
  for (BlockID block : p_graph.blocks()) {
    ss << p_ctx.perfectly_balanced_block_weight(block) << ", ";
  }
  ss << "\b\b]\n";
  return ss.str();
}

} // namespace kaminpar::shm::debug
