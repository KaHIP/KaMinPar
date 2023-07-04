/*******************************************************************************
 * Multilevel graph partitioning with direct k-way initial partitioning.
 *
 * @file:   kway_partitioner.cc
 * @author: Daniel Seemaier
 * @date:   25.10.2021
 ******************************************************************************/
#include "dkaminpar/partitioning/kway_multilevel.h"

#include "dkaminpar/coarsening/coarsener.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/factories.h"
#include "dkaminpar/graphutils/replicator.h"
#include "dkaminpar/metrics.h"

#include "kaminpar/datastructures/graph.h"
#include "kaminpar/metrics.h"

#include "common/console_io.h"
#include "common/strutils.h"
#include "common/timer.h"

namespace kaminpar::dist {
SET_DEBUG(false);

KWayMultilevelPartitioner::KWayMultilevelPartitioner(const DistributedGraph &graph, const Context &ctx)
    : _graph(graph),
      _ctx(ctx) {}

DistributedPartitionedGraph KWayMultilevelPartitioner::partition() {
  Coarsener coarsener(_graph, _ctx);

  const DistributedGraph *graph = &_graph;
  const bool is_root = mpi::get_comm_rank(_graph.communicator());

  ////////////////////////////////////////////////////////////////////////////////
  // Step 1: Coarsening
  ////////////////////////////////////////////////////////////////////////////////
  if (is_root) {
    cio::print_banner("Coarsening");
  }

  {
    SCOPED_TIMER("Coarsening");

    const GlobalNodeID threshold = (_ctx.simulate_singlethread ? 1 : _ctx.parallel.num_threads) *
                                   _ctx.partition.k * _ctx.coarsening.contraction_limit;
    while (graph->global_n() > threshold) {
      SCOPED_TIMER("Coarsening", std::string("Level ") + std::to_string(coarsener.level()));
      const GlobalNodeWeight max_cluster_weight = coarsener.max_cluster_weight();

      const DistributedGraph *c_graph = coarsener.coarsen_once();
      const bool converged = (graph == c_graph);

      if (!converged) {
        // Print statistics for coarse graph
        const std::string n_str = mpi::gather_statistics_str(c_graph->n(), c_graph->communicator());
        const std::string ghost_n_str =
            mpi::gather_statistics_str(c_graph->ghost_n(), c_graph->communicator());
        const std::string m_str = mpi::gather_statistics_str(c_graph->m(), c_graph->communicator());
        const std::string max_node_weight_str = mpi::gather_statistics_str<GlobalNodeWeight>(
            c_graph->max_node_weight(), c_graph->communicator()
        );

        // Machine readable
        LOG << "=> level=" << coarsener.level() << " "
            << "global_n=" << c_graph->global_n() << " "
            << "global_m=" << c_graph->global_m() << " "
            << "n=[" << n_str << "] "
            << "ghost_n=[" << ghost_n_str << "] "
            << "m=[" << m_str << "] "
            << "max_node_weight=[" << max_node_weight_str << "] "
            << "max_cluster_weight=" << max_cluster_weight;

        // Human readable
        LOG << "Level " << coarsener.level() << ":";
        graph::print_summary(*c_graph);

        graph = c_graph;
      } else if (converged) {
        LOG << "==> Coarsening converged";
        break;
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Step 2: Initial Partitioning
  ////////////////////////////////////////////////////////////////////////////////
  if (mpi::get_comm_rank(_graph.communicator()) == 0) {
    cio::print_banner("Initial Partitioning");
  }

  auto initial_partitioner = TIMED_SCOPE("Allocation") {
    return factory::create_initial_partitioner(_ctx);
  };

  START_TIMER("Initial Partitioning");
  auto shm_graph = graph::replicate_everywhere(*graph);
  shm::PartitionedGraph shm_p_graph{};
  if (_ctx.simulate_singlethread) {
    shm_p_graph = initial_partitioner->initial_partition(shm_graph, _ctx.partition);
    EdgeWeight best_cut = shm::metrics::edge_cut(shm_p_graph);

    for (std::size_t rep = 1; rep < _ctx.parallel.num_threads; ++rep) {
      auto partition = initial_partitioner->initial_partition(shm_graph, _ctx.partition);
      const auto cut = shm::metrics::edge_cut(partition);
      if (cut < best_cut) {
        best_cut = cut;
        shm_p_graph = std::move(partition);
      }
    }
  } else {
    shm_p_graph = initial_partitioner->initial_partition(shm_graph, _ctx.partition);
  }
  DistributedPartitionedGraph dist_p_graph =
      graph::distribute_best_partition(*graph, std::move(shm_p_graph));
  STOP_TIMER();

  KASSERT(
      graph::debug::validate_partition(dist_p_graph),
      "graph partition verification failed after initial partitioning",
      assert::heavy
  );

  const auto initial_cut = metrics::edge_cut(dist_p_graph);
  const auto initial_imbalance = metrics::imbalance(dist_p_graph);

  LOG << "Initial partition: cut=" << initial_cut << " imbalance=" << initial_imbalance;

  ////////////////////////////////////////////////////////////////////////////////
  // Step 3: Refinement
  ////////////////////////////////////////////////////////////////////////////////
  {
    SCOPED_TIMER("Uncoarsening");
    auto ref_p_ctx = _ctx.partition;
    ref_p_ctx.graph = std::make_unique<GraphContext>(dist_p_graph.graph(), ref_p_ctx);

    if (mpi::get_comm_rank(_graph.communicator()) == 0) {
      cio::print_banner("Refinement");
    }

    auto refiner_factory = TIMED_SCOPE("Allocation") {
      return factory::create_refiner(_ctx);
    };

    auto refine = [&](DistributedPartitionedGraph &p_graph) {
      SCOPED_TIMER("Refinement");
      LOG << "-> Refining partition ...";
      auto refiner = refiner_factory->create(p_graph, _ctx.partition);
      refiner->initialize();
      refiner->refine();
      KASSERT(
          graph::debug::validate_partition(p_graph),
          "graph partition verification failed after refinement",
          assert::heavy
      );
    };

    // special case: graph too small for multilevel, still run refinement
    if (_ctx.refinement.refine_coarsest_level) {
      SCOPED_TIMER("Uncoarsening", std::string("Level ") + std::to_string(coarsener.level()));
      ref_p_ctx.graph = std::make_unique<GraphContext>(dist_p_graph.graph(), ref_p_ctx);
      refine(dist_p_graph);

      // Output refinement statistics
      const auto current_cut = metrics::edge_cut(dist_p_graph);
      const auto current_imbalance = metrics::imbalance(dist_p_graph);

      LOG << "=> level=" << coarsener.level() << " cut=" << current_cut
          << " imbalance=" << current_imbalance;
    }

    // Uncoarsen and refine
    while (coarsener.level() > 0) {
      SCOPED_TIMER("Uncoarsening", std::string("Level ") + std::to_string(coarsener.level()));

      dist_p_graph = TIMED_SCOPE("Uncontraction") {
        return coarsener.uncoarsen_once(std::move(dist_p_graph));
      };
      ref_p_ctx.graph = std::make_unique<GraphContext>(dist_p_graph.graph(), ref_p_ctx);

      refine(dist_p_graph);

      // Output refinement statistics
      const auto current_cut = metrics::edge_cut(dist_p_graph);
      const auto current_imbalance = metrics::imbalance(dist_p_graph);

      LOG << "=> level=" << coarsener.level() << " cut=" << current_cut
          << " imbalance=" << current_imbalance;
    }
  }

  return dist_p_graph;
}
} // namespace kaminpar::dist
