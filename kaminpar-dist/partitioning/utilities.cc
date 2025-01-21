/*******************************************************************************
 * Utility functions for partitioning schemes.
 *
 * @file:   utilities.cc
 * @author: Daniel Seemaier
 * @date:   16.01.2023
 ******************************************************************************/
#include "kaminpar-dist/partitioning/utilities.h"

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/logger.h"
#include "kaminpar-dist/metrics.h"
#include "kaminpar-dist/timer.h"

#include "kaminpar-shm/partitioning/partition_utils.h"

namespace kaminpar::dist {

namespace {

SET_DEBUG(true);

}

PartitionContext create_refinement_context(
    const Context &input_ctx, const DistributedGraph &graph, const BlockID current_k, bool toplevel
) {
  const BlockID input_k = input_ctx.partition.k;

  std::vector<BlockWeight> max_block_weights(current_k);
  BlockID cur_fine_block = 0;
  for (BlockID coarse_block = 0; coarse_block < current_k; ++coarse_block) {
    const BlockID num = shm::partitioning::compute_final_k(coarse_block, current_k, input_k);
    const BlockID begin = cur_fine_block;
    const BlockID end = cur_fine_block + num;
    cur_fine_block += num;

    max_block_weights[coarse_block] =
        input_ctx.partition.total_unrelaxed_max_block_weights(begin, end);
  }

  PartitionContext new_p_ctx;
  new_p_ctx.setup(graph, std::move(max_block_weights), !toplevel);

  // @todo
  if (input_ctx.partition.has_epsilon()) {
    new_p_ctx.set_epsilon(input_ctx.partition.epsilon());
  }

  return new_p_ctx;
}

shm::PartitionContext create_initial_partitioning_context(
    const Context &input_ctx,
    const shm::Graph &graph,
    const BlockID current_block,
    const BlockID current_k,
    const BlockID desired_k,
    const bool toplevel
) {
  const BlockID k = (desired_k == input_ctx.partition.k)
                        ? shm::partitioning::compute_final_k(current_block, current_k, desired_k)
                        : desired_k / current_k;

  std::vector<shm::BlockWeight> max_block_weights(k);
  BlockID cur_begin =
      shm::partitioning::compute_first_sub_block(current_block, current_k, input_ctx.partition.k);

  for (BlockID b = 0; b < k; ++b) {
    const BlockID num_subblocks = [&]() -> BlockID {
      if (desired_k == input_ctx.partition.k) {
        return 1;
      } else {
        return shm::partitioning::compute_final_k(
            current_block * k + b, desired_k, input_ctx.partition.k
        );
      }
    }();

    max_block_weights[b] =
        input_ctx.partition.total_unrelaxed_max_block_weights(cur_begin, cur_begin + num_subblocks);
    cur_begin += num_subblocks;
  }

  DBG << "Requested shm::PartitionContext for " << current_k << " -> " << desired_k
      << " initial / recursive partitioning, thus splitting the block-induced graph with"
      << " n=" << graph.n() << " nodes and m=" << graph.m() << " edges into " << k
      << " sub-graphs with max_block_weights=[" << max_block_weights
      << "]; further relax weights: " << (!toplevel ? "yes" : "no");

  shm::PartitionContext p_ctx;
  p_ctx.setup(graph, std::move(max_block_weights), !toplevel);

  return p_ctx;
}

void print_input_graph(const DistributedGraph &graph, const bool verbose) {
  TIMER_BARRIER(graph.communicator());
  SCOPED_TIMER("Print graph statistics");

  LOG << "Input graph:";
  if (verbose) {
    print_extended_graph_summary(graph);
  } else {
    print_graph_summary(graph);
  }
  LOG;

  TIMER_BARRIER(graph.communicator());
}

void print_coarsened_graph(
    const DistributedGraph &graph,
    const int level,
    const GlobalNodeWeight max_cluster_weight,
    const bool verbose
) {
  TIMER_BARRIER(graph.communicator());
  SCOPED_TIMER("Print graph statistics");

  LOG << "Coarsening -> Level " << level << " [max cluster weight: " << max_cluster_weight << "]:";
  if (verbose) {
    print_extended_graph_summary(graph);
  } else {
    print_graph_summary(graph);
  }
  LOG;

  TIMER_BARRIER(graph.communicator());
}

void print_coarsening_converged() {
  LOG << "==> Coarsening converged";
  LOG;
}

void print_coarsening_terminated(const GlobalNodeID desired_num_nodes) {
  LOG << "==> Coarsening terminated with less than " << desired_num_nodes << " nodes";
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
  LOG << " Number of blocks: " << p_graph.k();
  LOG << " Cut:              " << cut;
  LOG << " Imbalance:        " << imbalance;
  LOG << " Feasible:         " << (feasible ? "yes" : "no");
}

} // namespace kaminpar::dist
