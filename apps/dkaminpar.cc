/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
// This must come first since it redefines output macros (LOG DBG etc)
// clang-format off
#include "dkaminpar/distributed_definitions.h"
// clang-format on

#include "apps.h"
#include "dkaminpar/algorithm/allgather_graph.h"
#include "dkaminpar/algorithm/distributed_local_graph_contraction.h"
#include "dkaminpar/application/arguments.h"
#include "dkaminpar/coarsening/distributed_local_label_propagation_coarsener.h"
#include "dkaminpar/distributed_context.h"
#include "dkaminpar/distributed_io.h"
#include "dkaminpar/refinement/distributed_label_propagation_refiner.h"
#include "dkaminpar/utility/distributed_metrics.h"
#include "kaminpar/definitions.h"
#include "kaminpar/partitioning_scheme/partitioning.h"
#include "kaminpar/utility/logger.h"
#include "kaminpar/utility/metrics.h"
#include "kaminpar/utility/random.h"

#include <fstream>
#include <mpi.h>
#include <partitioning_scheme/partitioning.h>

namespace dist = dkaminpar;
namespace shm = kaminpar;

// clang-format off
void sanitize_context(const dist::DContext &ctx) {
  ALWAYS_ASSERT(!std::ifstream(ctx.graph_filename) == false) << "Graph file cannot be read. Ensure that the file exists and is readable.";
  ALWAYS_ASSERT(ctx.partition.k >= 2) << "k must be at least 2.";
  ALWAYS_ASSERT(ctx.partition.epsilon > 0) << "Epsilon cannot be zero.";
}
// clang-format on

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  shm::print_identifier(argc, argv);

  {
    // Parse command line arguments
    dist::DContext ctx;
    try {
      ctx = dist::app::parse_options(argc, argv);
      sanitize_context(ctx);
    } catch (const std::runtime_error &e) { FATAL_ERROR << e.what(); }
    shm::Context shm_ctx = shm::create_default_context();

    // Initialize
    shm::Randomize::seed = ctx.seed;
    auto gc = shm::init_parallelism(ctx.parallel.num_threads);
    if (ctx.parallel.use_interleaved_numa_allocation) { shm::init_numa(); }

    MPI_Barrier(MPI_COMM_WORLD);

    // Load graph
    auto graph = dist::io::metis::read_node_balanced(ctx.graph_filename);
    MPI_Barrier(MPI_COMM_WORLD);
    graph.print_info();
    MPI_Barrier(MPI_COMM_WORLD);

    // Coarsen graph
    std::vector<dist::DistributedGraph> graph_hierarchy;
    std::vector<dist::scalable_vector<dist::DNodeID>> mapping_hierarchy;

    const dist::DistributedGraph *c_graph = &graph;
    while (c_graph->n() > 2 * shm_ctx.coarsening.contraction_limit) {
      const dist::DNodeWeight max_cluster_weight = shm::compute_max_cluster_weight(c_graph->global_n(),
                                                                                   c_graph->total_node_weight(),
                                                                                   shm_ctx.partition,
                                                                                   shm_ctx.coarsening);
      dist::DistributedLocalLabelPropagationClustering coarsener(c_graph->n(), 0.5, ctx.coarsening.lp);
      auto &clustering = coarsener.cluster(*c_graph, max_cluster_weight, ctx.coarsening.lp.num_iterations);

      auto [contracted_graph, mapping, mem] = dist::graph::contract_locally(*c_graph, clustering);
      const bool converged = contracted_graph.global_n() == c_graph->global_n();
      graph_hierarchy.push_back(std::move(contracted_graph));
      mapping_hierarchy.push_back(std::move(mapping));
      c_graph = &graph_hierarchy.back();

      LOG << "=> n=" << c_graph->global_n() << " m=" << c_graph->global_m();
      if (converged) {
        LOG << "==> Coarsening converged";
        break;
      }
    }

    // initial partitioning
    shm::Graph shm_graph = dist::graph::allgather(*c_graph);
    shm_ctx.refinement.lp.num_iterations = 1;
    shm_ctx.partition.k = ctx.partition.k;
    shm_ctx.partition.epsilon = ctx.partition.epsilon;
    shm_ctx.setup(shm_graph);

    auto shm_p_graph = shm::partitioning::partition(shm_graph, shm_ctx);
    DLOG << "Obtained " << shm_ctx.partition.k << "-way partition with cut=" << shm::metrics::edge_cut(shm_p_graph)
         << " and imbalance=" << shm::metrics::imbalance(shm_p_graph);

    dist::DistributedPartitionedGraph dist_p_graph = dist::graph::create_from_best_partition(graph,
                                                                                             std::move(shm_p_graph));
    dist::debug::validate_partition_state(dist_p_graph);

    LOG << "Initial partition: cut=" << dist::metrics::edge_cut(dist_p_graph)
        << " imbalance=" << dist::metrics::imbalance(dist_p_graph);

    auto refine = [&](dist::DistributedPartitionedGraph &p_graph) {
      dist::DistributedLabelPropagationRefiner<dist::DBlockID, dist::DBlockWeight>
          lp(ctx.refinement.lp, &p_graph, static_cast<dist::DBlockID>(ctx.partition.k),
             static_cast<dist::DBlockWeight>(shm_ctx.partition.max_block_weight(0)));
      for (std::size_t i = 0; i < ctx.refinement.lp.num_iterations; ++i) { lp.perform_iteration(); }
    };

    // Uncoarsen and refine
    refine(dist_p_graph);
    while (!graph_hierarchy.empty()) {
      // (1) Uncoarsen graph
      auto mapping = std::move(mapping_hierarchy.back());
      graph_hierarchy.pop_back();
      mapping_hierarchy.pop_back(); // destroy graph wrapped in dist_p_graph, but partition access is still ok

      // create partition for new coarsest graph
      const auto *current_graph = graph_hierarchy.empty() ? &graph : &graph_hierarchy.back();
      dist::scalable_vector<dist::DBlockID> partition(current_graph->total_n());
      current_graph->pfor_all_nodes([&](const dist::DNodeID u) {
        partition[u] = dist_p_graph.block(mapping[u]);
      });
      dist_p_graph = dist::DistributedPartitionedGraph{current_graph, ctx.partition.k, std::move(partition),
                                                       std::move(dist_p_graph.take_block_weights())};

      // (2) Refine
      refine(dist_p_graph);

      DLOG << "Cut after LP: cut=" << dist::metrics::edge_cut(dist_p_graph)
           << " imbalance=" << dist::metrics::imbalance(dist_p_graph);
    }
  }

  MPI_Finalize();
  return 0;
}