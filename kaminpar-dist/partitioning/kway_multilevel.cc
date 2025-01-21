/*******************************************************************************
 * Multilevel graph partitioning with direct k-way initial partitioning.
 *
 * @file:   kway_partitioner.cc
 * @author: Daniel Seemaier
 * @date:   25.10.2021
 ******************************************************************************/
#include "kaminpar-dist/partitioning/kway_multilevel.h"

#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/coarsening/coarsener.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/factories.h"
#include "kaminpar-dist/graphutils/replicator.h"
#include "kaminpar-dist/logger.h"
#include "kaminpar-dist/metrics.h"
#include "kaminpar-dist/partitioning/utilities.h"
#include "kaminpar-dist/timer.h"

#include "kaminpar-shm/metrics.h"

namespace kaminpar::dist {

namespace {

SET_DEBUG(false);

}

KWayMultilevelPartitioner::KWayMultilevelPartitioner(
    const DistributedGraph &input_graph, const Context &input_ctx
)
    : _input_graph(input_graph),
      _input_ctx(input_ctx) {}

DistributedPartitionedGraph KWayMultilevelPartitioner::partition() {
  auto coarsener = factory::create_coarsener(_input_ctx);
  coarsener->initialize(&_input_graph);

  const DistributedGraph *graph = &_input_graph;

  ////////////////////////////////////////////////////////////////////////////////
  // Step 1: Coarsening
  ////////////////////////////////////////////////////////////////////////////////
  {
    SCOPED_TIMER("Coarsening");

    const GlobalNodeID threshold =
        (_input_ctx.partitioning.simulate_singlethread ? 1 : _input_ctx.parallel.num_threads) *
        _input_ctx.partition.k * _input_ctx.coarsening.contraction_limit;
    while (graph->global_n() > threshold) {
      SCOPED_TIMER("Coarsening", std::string("Level ") + std::to_string(coarsener->level()));
      const bool converged = !coarsener->coarsen();

      if (!converged) {
        const DistributedGraph &c_graph = coarsener->current();

        // Print statistics for coarse graph
        const std::string n_str = mpi::gather_statistics_str(c_graph.n(), c_graph.communicator());
        const std::string ghost_n_str =
            mpi::gather_statistics_str(c_graph.ghost_n(), c_graph.communicator());
        const std::string m_str = mpi::gather_statistics_str(c_graph.m(), c_graph.communicator());
        const std::string max_node_weight_str = mpi::gather_statistics_str<GlobalNodeWeight>(
            c_graph.max_node_weight(), c_graph.communicator()
        );

        // Machine readable
        LOG << "=> level=" << coarsener->level() << " "
            << "global_n=" << c_graph.global_n() << " "
            << "global_m=" << c_graph.global_m() << " "
            << "n=[" << n_str << "] "
            << "ghost_n=[" << ghost_n_str << "] "
            << "m=[" << m_str << "] "
            << "max_node_weight=[" << max_node_weight_str << "]";

        // Human readable
        LOG << "Level " << coarsener->level() << ":";
        print_graph_summary(c_graph);

        graph = &c_graph;
      } else if (converged) {
        LOG << "==> Coarsening converged";
        break;
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Step 2: Initial Partitioning
  ////////////////////////////////////////////////////////////////////////////////
  DistributedPartitionedGraph dist_p_graph = TIMED_SCOPE("Initial partitioning") {
    auto initial_partitioner = TIMED_SCOPE("Allocation") {
      return factory::create_initial_partitioner(_input_ctx);
    };

    const PartitionContext p_ctx =
        create_refinement_context(_input_ctx, *graph, _input_ctx.partition.k, coarsener->empty());

    shm::Graph shm_graph = replicate_graph_everywhere(*graph);
    shm::PartitionedGraph shm_p_graph = initial_partitioner->initial_partition(shm_graph, p_ctx);

    if (_input_ctx.partitioning.simulate_singlethread) {
      shm::EdgeWeight shm_cut = shm::metrics::edge_cut(shm_p_graph);
      for (std::size_t rep = 1; rep < _input_ctx.parallel.num_threads; ++rep) {
        shm::PartitionedGraph next_shm_p_graph =
            initial_partitioner->initial_partition(shm_graph, p_ctx);
        const shm::EdgeWeight next_cut = shm::metrics::edge_cut(next_shm_p_graph);

        if (next_cut < shm_cut) {
          shm_cut = next_cut;
          shm_p_graph = std::move(next_shm_p_graph);
        }
      }
    }

    return distribute_best_partition(*graph, std::move(shm_p_graph));
  };

  KASSERT(
      debug::validate_partition(dist_p_graph),
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

    auto refiner_factory = TIMED_SCOPE("Allocation") {
      return factory::create_refiner(_input_ctx);
    };

    auto refine = [&](DistributedPartitionedGraph &p_graph) {
      SCOPED_TIMER("Refinement");
      LOG << "-> Refining partition ...";
      const PartitionContext ref_p_ctx = create_refinement_context(
          _input_ctx, dist_p_graph.graph(), _input_ctx.partition.k, coarsener->empty()
      );

      auto refiner = refiner_factory->create(p_graph, ref_p_ctx);
      refiner->initialize();
      refiner->refine();

      KASSERT(
          debug::validate_partition(p_graph),
          "graph partition verification failed after refinement",
          assert::heavy
      );
    };

    // Special case: graph too small for multilevel, still run refinement
    if (_input_ctx.refinement.refine_coarsest_level) {
      SCOPED_TIMER("Uncoarsening", std::string("Level ") + std::to_string(coarsener->level()));
      refine(dist_p_graph);

      // Output refinement statistics
      const GlobalEdgeWeight current_cut = metrics::edge_cut(dist_p_graph);
      const double current_imbalance = metrics::imbalance(dist_p_graph);

      LOG << "=> level=" << coarsener->level() << " cut=" << current_cut
          << " imbalance=" << current_imbalance;
    }

    // Uncoarsen and refine
    while (coarsener->level() > 0) {
      SCOPED_TIMER("Uncoarsening", std::string("Level ") + std::to_string(coarsener->level()));

      dist_p_graph = TIMED_SCOPE("Uncontraction") {
        return coarsener->uncoarsen(std::move(dist_p_graph));
      };

      refine(dist_p_graph);

      // Output refinement statistics
      const GlobalEdgeWeight current_cut = metrics::edge_cut(dist_p_graph);
      const double current_imbalance = metrics::imbalance(dist_p_graph);

      LOG << "=> level=" << coarsener->level() << " cut=" << current_cut
          << " imbalance=" << current_imbalance;
    }
  }

  return dist_p_graph;
}

} // namespace kaminpar::dist
