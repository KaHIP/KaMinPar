/*******************************************************************************
 * Collection of debug utilities.
 *
 * @file:   debug.cc
 * @author: Daniel Seemaier
 * @date:   31.03.2023
 ******************************************************************************/
#include "kaminpar-dist/debug.h"

#include <fstream>
#include <string>

#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"

namespace kaminpar::dist::debug {
namespace {
std::string generate_filename(
    const DistributedGraph &graph, const DebugContext &d_ctx, const std::string &suffix
) {
  std::stringstream filename_ss;

  if (d_ctx.graph_filename.empty()) {
    filename_ss << "undefined_n" << graph.global_n() << "_m" << graph.global_m();
  } else {
    filename_ss << d_ctx.graph_filename;
  }
  filename_ss << "." << suffix;
  return filename_ss.str();
}
} // namespace

void write_coarsest_graph(const DistributedGraph &graph, const DebugContext &d_ctx) {
  if (d_ctx.save_coarsest_graph) {
    write_metis_graph(generate_filename(graph, d_ctx, "coarsest.metis"), graph);
  }
}

void write_metis_graph(const std::string &filename, const DistributedGraph &graph) {
  const PEID size = mpi::get_comm_size(graph.communicator());
  const PEID rank = mpi::get_comm_rank(graph.communicator());

  if (rank == 0) {
    std::ofstream out(filename, std::ios::trunc);
    out << graph.global_n() << " " << graph.global_m() / 2 << " ";
    if (graph.is_node_weighted()) {
      out << graph.is_node_weighted() << graph.is_edge_weighted();
    } else if (graph.is_edge_weighted()) {
      out << graph.is_edge_weighted();
    }
    out << "\n";
  }

  mpi::barrier(graph.communicator());

  for (PEID pe = 0; pe < size; ++pe) {
    if (pe == rank) {
      std::ofstream out(filename, std::ios::app);

      for (const NodeID lu : graph.nodes()) {
        if (graph.is_node_weighted()) {
          out << graph.node_weight(lu) << " ";
        }
        for (const auto &[e, lv] : graph.neighbors(lu)) {
          out << graph.local_to_global_node(lv) + 1 << " ";
          if (graph.is_edge_weighted()) {
            out << graph.edge_weight(e) << " ";
          }
        }
        out << "\n";
      }
    }

    mpi::barrier(graph.communicator());
  }
}

void write_coarsest_partition(
    const DistributedPartitionedGraph &p_graph, const DebugContext &d_ctx
) {
  if (d_ctx.save_coarsest_partition) {
    write_partition(generate_filename(p_graph.graph(), d_ctx, ".coarsest.part"), p_graph, true);
  }
}

void write_partition(
    const std::string &filename,
    const DistributedPartitionedGraph &p_graph,
    const bool use_original_node_order
) {
  const PEID size = mpi::get_comm_size(p_graph.communicator());
  const PEID rank = mpi::get_comm_rank(p_graph.communicator());

  if (rank == 0) {
    std::ofstream out(filename, std::ios::trunc);
  }

  mpi::barrier(p_graph.communicator());

  for (PEID pe = 0; pe < size; ++pe) {
    if (pe == rank) {
      std::ofstream out(filename, std::ios::app);

      for (const NodeID lu : p_graph.nodes()) {
        const NodeID mapped =
            p_graph.permuted() && use_original_node_order ? p_graph.map_original_node(lu) : lu;
        out << p_graph.block(mapped) << "\n";
      }
    }

    mpi::barrier(p_graph.communicator());
  }
}
} // namespace kaminpar::dist::debug
