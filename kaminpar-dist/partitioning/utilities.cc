/*******************************************************************************
 * Utility functions for partitioning schemes.
 *
 * @file:   utilities.cc
 * @author: Daniel Seemaier
 * @date:   16.01.2023
 ******************************************************************************/
#include "kaminpar-dist/partitioning/utilities.h"

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/metrics.h"
#include "kaminpar-dist/timer.h"

namespace kaminpar::dist {
void print_input_graph(const DistributedGraph &graph) {
  TIMER_BARRIER(graph.communicator());
  SCOPED_TIMER("Print graph statistics");

  LOG << "Input graph:";
  print_graph_summary(graph);
  LOG;

  TIMER_BARRIER(graph.communicator());
}

void print_coarsened_graph(
    const DistributedGraph &graph, const int level, const GlobalNodeWeight max_cluster_weight
) {
  TIMER_BARRIER(graph.communicator());
  SCOPED_TIMER("Print graph statistics");

  LOG << "Coarsening -> Level " << level << ":";
  print_graph_summary(graph);
  LOG << "    <= " << max_cluster_weight;
  LOG;

  TIMER_BARRIER(graph.communicator());
}

void print_coarsening_converged() {
  LOG << "==> Coarsening converged.";
  LOG;
}

void print_coarsening_terminated(const GlobalNodeID desired_num_nodes) {
  LOG << "==> Coarsening terminated with less than " << desired_num_nodes << " nodes.";
  LOG;
}

void print_initial_partitioning_result(
    const DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
) {
  TIMER_BARRIER(p_graph.communicator());
  SCOPED_TIMER("Print partition statistics");

  const GlobalEdgeWeight cut = metrics::edge_cut(p_graph);
  const double imbalance = metrics::imbalance(p_graph);
  const bool feasible = metrics::is_feasible(p_graph, p_ctx);

  LOG << "Initial partition:";
  LOG << "  Number of blocks: " << p_graph.k();
  LOG << "  Cut:              " << cut;
  LOG << "  Imbalance:        " << imbalance;
  LOG << "  Feasible:         " << (feasible ? "yes" : "no");
}
} // namespace kaminpar::dist
