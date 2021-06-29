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
#include "dkaminpar/algorithm/distributed_label_propagation.h"
#include "dkaminpar/application/arguments.h"
#include "dkaminpar/distributed_context.h"
#include "dkaminpar/distributed_io.h"
#include "dkaminpar/utility/distributed_metrics.h"
#include "kaminpar/definitions.h"
#include "kaminpar/partitioning_scheme/partitioning.h"
#include "kaminpar/utility/logger.h"
#include "kaminpar/utility/metrics.h"
#include "kaminpar/utility/random.h"

#include <fstream>
#include <mpi.h>

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

    // ... coarsening ...

    // initial partitioning
    shm::Graph shm_graph = dist::graph::allgather(graph);
    shm::Context shm_ctx = shm::create_default_context();

    // disable refinement
    shm_ctx.initial_partitioning.refinement.algorithm = shm::RefinementAlgorithm::NOOP;
    shm_ctx.refinement.algorithm = shm::RefinementAlgorithm::NOOP;

    shm_ctx.partition.k = ctx.partition.k;
    shm_ctx.partition.epsilon = ctx.partition.epsilon;
    shm_ctx.setup(shm_graph);

    shm::StaticArray<shm::BlockID> part(shm_graph.n());
    shm_graph.pfor_nodes([&](const shm::NodeID u) { part[u] = u % shm_ctx.partition.k; });

//    auto shm_p_graph = shm::partitioning::partition(shm_graph, shm_ctx);
    shm::PartitionedGraph shm_p_graph{shm_graph, shm_ctx.partition.k, std::move(part)};
    DLOG << "Obtained " << shm_ctx.partition.k << "-way partition with cut=" << shm::metrics::edge_cut(shm_p_graph)
         << " and imbalance=" << shm::metrics::imbalance(shm_p_graph);

    dist::DistributedPartitionedGraph dist_p_graph = dist::graph::create_from_best_partition(graph,
                                                                                             std::move(shm_p_graph));

    DLOG << "On distributed graph: cut=" << dist::metrics::edge_cut(dist_p_graph)
         << " imbalance=" << dist::metrics::imbalance(dist_p_graph);

    // Run LP
    dist::DistributedLabelPropagation<dist::DBlockID, dist::DBlockWeight>
        lp(&dist_p_graph, static_cast<dist::DBlockID>(ctx.partition.k),
           static_cast<dist::DBlockWeight>(shm_ctx.partition.max_block_weight(0)));
    lp.perform_iteration();

    DLOG << "Cut after LP: cut=" << dist::metrics::edge_cut(dist_p_graph)
         << " imbalance=" << dist::metrics::imbalance(dist_p_graph);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  LOG << "Finalize next";
  MPI_Finalize();
  return 0;
}