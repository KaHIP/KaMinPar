/*******************************************************************************
 * Deep multilevel graph partitioning with direct k-way initial partitioning.
 *
 * @file:   deep_multilevel_partitioner.cc
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 ******************************************************************************/
#include "kaminpar-dist/partitioning/deep_multilevel.h"

#include <iomanip>

#include <mpi.h>

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/debug.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/factories.h"
#include "kaminpar-dist/graphutils/replicator.h"
#include "kaminpar-dist/graphutils/subgraph_extractor.h"
#include "kaminpar-dist/logger.h"
#include "kaminpar-dist/metrics.h"
#include "kaminpar-dist/partitioning/utilities.h"
#include "kaminpar-dist/timer.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/metrics.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/math.h"

namespace kaminpar::dist {

namespace {

SET_DEBUG(false);

}

DeepMultilevelPartitioner::DeepMultilevelPartitioner(
    const DistributedGraph &input_graph, const Context &input_ctx
)
    : _input_graph(input_graph),
      _input_ctx(input_ctx) {
  _coarseners.emplace(factory::create_coarsener(_input_ctx));
  _coarseners.top()->initialize(&_input_graph);
}

DistributedPartitionedGraph DeepMultilevelPartitioner::partition() {
  const DistributedGraph *graph = &_input_graph;
  auto *coarsener = get_current_coarsener();
  bool converged = false;

  print_input_graph(_input_graph, _print_graph_stats);

  /*
   * Coarsening
   */
  const BlockID first_step_k = std::min<BlockID>(
      _input_ctx.partition.k,
      _input_ctx.partitioning.initial_k > 0 ? _input_ctx.partitioning.initial_k : 2
  );
  const GlobalNodeID desired_num_nodes =
      (_input_ctx.partitioning.simulate_singlethread ? 1 : _input_ctx.parallel.num_threads) *
      _input_ctx.coarsening.contraction_limit * first_step_k;
  const PEID initial_rank = mpi::get_comm_rank(_input_graph.communicator());
  const PEID initial_size = mpi::get_comm_size(_input_graph.communicator());
  PEID current_num_pes = initial_size;

  int level = 0;

  START_HEAP_PROFILER("Coarsening");
  while (!converged && graph->global_n() > desired_num_nodes) {
    SCOPED_HEAP_PROFILER("Level", std::to_string(coarsener->level()));
    SCOPED_TIMER("Coarsening");

    // Replicate graph and split PEs when the graph becomes too small
    const BlockID num_blocks_on_this_level =
        math::ceil2(graph->global_n() / _input_ctx.coarsening.contraction_limit);

    if (_input_ctx.partitioning.enable_pe_splitting && current_num_pes > 1 &&
        num_blocks_on_this_level < static_cast<BlockID>(current_num_pes)) {
      const PEID num_replications = current_num_pes / num_blocks_on_this_level;

      LOG << "Current graph (" << graph->global_n()
          << " nodes) is too small for the available parallelism (" << current_num_pes
          << "): duplicating the graph " << num_replications << " times";
      LOG;

      _replicated_graphs.push_back(replicate_graph(*graph, num_replications));
      _coarseners.emplace(factory::create_coarsener(_input_ctx));
      _coarseners.top()->initialize(&_replicated_graphs.back());

      graph = &_replicated_graphs.back();
      coarsener = get_current_coarsener();

      current_num_pes = mpi::get_comm_size(graph->communicator());
    }

    // Coarsen graph
    const GlobalNodeWeight max_cluster_weight = coarsener->max_cluster_weight();
    converged = !coarsener->coarsen();
    const DistributedGraph *c_graph = &coarsener->current();

    if (!converged) {
      ++level;
      print_coarsened_graph(coarsener->current(), level, max_cluster_weight, _print_graph_stats);
      if (c_graph->global_n() <= desired_num_nodes) {
        print_coarsening_terminated(desired_num_nodes);
      }
    } else {
      print_coarsening_converged();
    }

    graph = c_graph;
  }
  STOP_HEAP_PROFILER();
  TIMER_BARRIER(_input_graph.communicator());

  /*
   * Initial Partitioning
   */
  START_TIMER("Initial partitioning");
  START_HEAP_PROFILER("Initial partitioning");
  auto initial_partitioner = TIMED_SCOPE("Allocation") {
    SCOPED_HEAP_PROFILER("Allocation");
    return factory::create_initial_partitioner(_input_ctx);
  };

  shm::Graph shm_graph = replicate_graph_everywhere(*graph);
  const shm::PartitionContext shm_p_ctx =
      create_initial_partitioning_context(_input_ctx, shm_graph, 0, 1, first_step_k);
  shm::PartitionedGraph shm_p_graph = initial_partitioner->initial_partition(shm_graph, shm_p_ctx);

  if (_input_ctx.partitioning.simulate_singlethread) {
    shm::EdgeWeight shm_cut = shm::metrics::edge_cut(shm_p_graph);

    for (std::size_t rep = 1; rep < _input_ctx.parallel.num_threads; ++rep) {
      shm::PartitionedGraph next_shm_p_graph =
          initial_partitioner->initial_partition(shm_graph, shm_p_ctx);
      const shm::EdgeWeight next_cut = shm::metrics::edge_cut(next_shm_p_graph);

      if (next_cut < shm_cut) {
        shm_cut = next_cut;
        shm_p_graph = std::move(next_shm_p_graph);
      }
    }
  }

  DistributedPartitionedGraph dist_p_graph =
      distribute_best_partition(*graph, std::move(shm_p_graph));

  KASSERT(
      debug::validate_partition(dist_p_graph),
      "invalid partition after initial partitioning",
      assert::heavy
  );

  print_initial_partitioning_result(
      dist_p_graph,
      create_refinement_context(
          _input_ctx, dist_p_graph.graph(), dist_p_graph.k(), &dist_p_graph.graph() == &_input_graph
      )
  );

  STOP_HEAP_PROFILER();
  STOP_TIMER();
  TIMER_BARRIER(_input_graph.communicator());

  // Only store coarsest graph + partition of PE group 0
  if (initial_rank < current_num_pes) {
    debug::write_coarsest_graph(*graph, _input_ctx.debug);
    debug::write_coarsest_partition(dist_p_graph, _input_ctx.debug);
  }

  /*
   * Uncoarsening and Refinement
   */
  START_TIMER("Uncoarsening");
  START_HEAP_PROFILER("Uncoarsening");
  auto refiner_factory = TIMED_SCOPE("Allocation") {
    SCOPED_HEAP_PROFILER("Allocation");
    return factory::create_refiner(_input_ctx);
  };

  auto run_refinement = [&](DistributedPartitionedGraph &p_graph,
                            const PartitionContext &ref_p_ctx) {
    TIMER_BARRIER(p_graph.communicator());
    START_TIMER("Refinement");
    START_HEAP_PROFILER("Refinement");

    auto refiner = refiner_factory->create(p_graph, ref_p_ctx);
    refiner->initialize();
    refiner->refine();

    STOP_HEAP_PROFILER();
    STOP_TIMER();

    KASSERT(
        debug::validate_partition(p_graph),
        "inconsistent partition after running refinement",
        assert::heavy
    );
  };

  auto extend_partition = [&](DistributedPartitionedGraph &p_graph,
                              const bool almost_toplevel = false,
                              const std::string &prefix = " ") -> PartitionContext {
    SCOPED_HEAP_PROFILER("Extending partition");

    BlockID desired_k = std::min<BlockID>(
        _input_ctx.partition.k,
        math::ceil2(dist_p_graph.global_n() / _input_ctx.coarsening.contraction_limit)
    );

    // If we (almost) work on the input graph, extend to final number of blocks
    if (_input_graph.global_n() == p_graph.global_n() ||
        (_input_ctx.partitioning.avoid_toplevel_bipartitioning && almost_toplevel &&
         _input_graph.global_n() >
             2 * _input_ctx.partition.k * _input_ctx.coarsening.contraction_limit)) {
      desired_k = _input_ctx.partition.k;
    }

    PartitionContext ref_p_ctx;
    if (dist_p_graph.k() == desired_k) {
      ref_p_ctx = create_refinement_context(
          _input_ctx, dist_p_graph.graph(), desired_k, &p_graph.graph() == &_input_graph
      );
    }

    while (dist_p_graph.k() < desired_k) {
      const BlockID next_k =
          _input_ctx.partitioning.extension_k > 0
              ? std::min<BlockID>(desired_k, dist_p_graph.k() * _input_ctx.partitioning.extension_k)
              : desired_k;

      LOG << prefix << "Extending partition from " << dist_p_graph.k() << " blocks to " << next_k
          << " blocks";

      // Extract blocks
      auto block_extraction_result =
          graph::extract_and_scatter_block_induced_subgraphs(dist_p_graph);
      const std::vector<shm::Graph> &subgraphs = block_extraction_result.subgraphs;

      // Partition block-induced subgraphs
      TIMER_BARRIER(dist_p_graph.communicator());
      START_TIMER("Initial partitioning");
      std::vector<shm::PartitionedGraph> p_subgraphs;

      const PEID size = mpi::get_comm_size(dist_p_graph.communicator());
      const PEID rank = mpi::get_comm_rank(dist_p_graph.communicator());
      const graph::BlockExtractionOffsets offsets(size, dist_p_graph.k());

      for (std::size_t i = 0; i < subgraphs.size(); ++i) {
        const BlockID block = offsets.first_block_on_pe(rank) + i;
        const shm::Graph &shm_graph = subgraphs[i];
        const shm::PartitionContext shm_p_ctx = create_initial_partitioning_context(
            _input_ctx, shm_graph, block, dist_p_graph.k(), desired_k
        );

        p_subgraphs.push_back(initial_partitioner->initial_partition(shm_graph, shm_p_ctx));
      }
      STOP_TIMER();

      // Project subgraph partitions onto dist_p_graph
      dist_p_graph = graph::copy_subgraph_partitions(
          std::move(dist_p_graph), p_subgraphs, block_extraction_result, next_k
      );

      ref_p_ctx = create_refinement_context(
          _input_ctx, dist_p_graph.graph(), next_k, &p_graph.graph() == &_input_graph
      );

      // Print statistics
      TIMER_BARRIER(dist_p_graph.communicator());
      START_TIMER("Print partition statistics");

      const GlobalEdgeWeight cut = metrics::edge_cut(dist_p_graph);
      const double imbalance = metrics::imbalance(dist_p_graph);
      const bool feasible = metrics::is_feasible(dist_p_graph, ref_p_ctx);

      LOG << prefix << " Cut:       " << cut;
      LOG << prefix << " Imbalance: " << std::setprecision(3) << imbalance;
      LOG << prefix << " Feasible:  " << (feasible ? "yes" : "no");

      STOP_TIMER();

      if (dist_p_graph.k() < desired_k) {
        LOG << prefix << "Running refinement on " << dist_p_graph.k() << " blocks";
        run_refinement(dist_p_graph, ref_p_ctx);

        START_TIMER("Print partition statistics");

        const GlobalEdgeWeight cut = metrics::edge_cut(dist_p_graph);
        const double imbalance = metrics::imbalance(dist_p_graph);
        const bool feasible = metrics::is_feasible(dist_p_graph, ref_p_ctx);

        LOG << prefix << " Cut:       " << cut;
        LOG << prefix << " Imbalance: " << imbalance;
        LOG << prefix << " Feasible:  " << (feasible ? "yes" : "no");

        STOP_TIMER();
      }
    }

    return ref_p_ctx;
  };

  // Uncoarsen, partition blocks and refine
  while (_coarseners.size() > 1 || (!_coarseners.empty() && coarsener->level() > 0)) {
    SCOPED_HEAP_PROFILER("Level", std::to_string(coarsener->level()));
    --level;

    LOG;
    if (level > 0) {
      LOG << "Uncoarsening -> Level " << level << ":";
    } else {
      LOG << "Toplevel:";
    }

    // Join split PE groups and use best partition
    if (coarsener->level() == 0) {
      LOG << " Joining split PE groups";

      KASSERT(!_coarseners.empty());
      _coarseners.pop();
      coarsener = get_current_coarsener();

      const DistributedGraph *new_graph = &coarsener->current();
      dist_p_graph = distribute_best_partition(*new_graph, std::move(dist_p_graph));

      _replicated_graphs.pop_back();
    }

    if (_input_ctx.partitioning.avoid_toplevel_bipartitioning && level == 0) {
      extend_partition(dist_p_graph, true);
    }

    // Uncoarsen graph
    // If we replicated early, we might already be on the finest level
    if (coarsener->level() > 0) {
      const GlobalNodeID prev_n = dist_p_graph.global_n();
      const GlobalEdgeID prev_m = dist_p_graph.global_m();
      dist_p_graph = coarsener->uncoarsen(std::move(dist_p_graph));

      LOG << " Uncoarsening graph: " << prev_n << " nodes, " << prev_m << " edges -> "
          << dist_p_graph.global_n() << " nodes, " << dist_p_graph.global_m() << " edges";
    }

    // Destroy coarsener before we run refinement on the finest level
    if (_coarseners.size() == 1 && coarsener->level() == 0) {
      _coarseners.pop();
    }

    // Extend partition
    const PartitionContext ref_p_ctx = extend_partition(dist_p_graph);

    // Run refinement
    LOG << " Running refinement on " << dist_p_graph.k() << " blocks";
    run_refinement(dist_p_graph, ref_p_ctx);

    // Output statistics
    TIMER_BARRIER(dist_p_graph.communicator());
    START_TIMER("Print partition statistics");

    const auto cut = metrics::edge_cut(dist_p_graph);
    const auto imbalance = metrics::imbalance(dist_p_graph);
    const bool feasible = metrics::is_feasible(dist_p_graph, ref_p_ctx);

    LOG << "  Cut:       " << cut;
    LOG << "  Imbalance: " << imbalance;
    LOG << "  Feasible:  " << (feasible ? "yes" : "no");

    STOP_TIMER();
  }

  // Extend partition if we have not already reached the desired number of
  // blocks This should only be used to cover the special case where the input
  // graph is too small for coarsening
  if (dist_p_graph.k() != _input_ctx.partition.k) {
    LOG;
    LOG << "Toplevel:";

    // Extend partition
    const PartitionContext ref_p_ctx = extend_partition(dist_p_graph);

    // Run refinement
    LOG << " Running balancing and local search on " << dist_p_graph.k() << " blocks";
    run_refinement(dist_p_graph, ref_p_ctx);

    // Output statistics
    TIMER_BARRIER(dist_p_graph.communicator());
    START_TIMER("Print partition statistics");

    const GlobalEdgeWeight cut = metrics::edge_cut(dist_p_graph);
    const double imbalance = metrics::imbalance(dist_p_graph);
    const bool feasible = metrics::is_feasible(dist_p_graph, ref_p_ctx);

    LOG << " Cut:       " << cut;
    LOG << " Imbalance: " << imbalance;
    LOG << " Feasible:  " << (feasible ? "yes" : "no");

    STOP_TIMER();
  }
  STOP_HEAP_PROFILER();
  STOP_TIMER();
  TIMER_BARRIER(_input_graph.communicator());

  return dist_p_graph;
}

Coarsener *DeepMultilevelPartitioner::get_current_coarsener() {
  KASSERT(!_coarseners.empty());
  return _coarseners.top().get();
}

const Coarsener *DeepMultilevelPartitioner::get_current_coarsener() const {
  KASSERT(!_coarseners.empty());
  return _coarseners.top().get();
}

} // namespace kaminpar::dist
