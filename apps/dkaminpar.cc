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
#include "dkaminpar/algorithm/distributed_graph_contraction.h"
#include "dkaminpar/application/arguments.h"
#include "dkaminpar/distributed_context.h"
#include "dkaminpar/distributed_io.h"
#include "dkaminpar/partitioning_scheme/partitioning.h"
#include "dkaminpar/utility/distributed_metrics.h"
#include "kaminpar/definitions.h"
#include "kaminpar/utility/logger.h"
#include "kaminpar/utility/random.h"
#include "kaminpar/utility/timer.h"

#include <fstream>
#include <mpi.h>

namespace dist = dkaminpar;
namespace shm = kaminpar;

// clang-format off
void sanitize_context(const dist::Context &ctx) {
  ALWAYS_ASSERT(!std::ifstream(ctx.graph_filename) == false) << "Graph file cannot be read. Ensure that the file exists and is readable: " << ctx.graph_filename;
  ALWAYS_ASSERT(ctx.partition.k >= 2) << "k must be at least 2.";
  ALWAYS_ASSERT(ctx.partition.epsilon > 0) << "Epsilon must be greater than zero.";
}
// clang-format on

void print_statistics(const dist::DistributedPartitionedGraph &p_graph, const dist::Context &ctx) {
  const auto edge_cut = dist::metrics::edge_cut(p_graph);
  const auto imbalance = dist::metrics::imbalance(p_graph);
  const auto feasible = dist::metrics::is_feasible(p_graph, ctx.partition);

  LOG << "RESULT cut=" << edge_cut << " imbalance=" << imbalance << " feasible=" << feasible << " k=" << p_graph.k();
  if (dist::mpi::get_comm_rank(MPI_COMM_WORLD) == 0 && !ctx.quiet) {
    shm::Timer::global().print_machine_readable(std::cout);
  }
  LOG;
  if (dist::mpi::get_comm_rank(MPI_COMM_WORLD) == 0 && !ctx.quiet) {
    shm::Timer::global().print_human_readable(std::cout);
  }
  LOG;
  LOG << "-> k=" << p_graph.k();
  LOG << "-> cut=" << edge_cut;
  LOG << "-> imbalance=" << imbalance;
  LOG << "-> feasible=" << feasible;
  if (p_graph.k() <= 512) {
    LOG << "-> block_weights:";
    LOG << shm::logger::TABLE << p_graph.block_weights();
  }

  if (p_graph.k() != ctx.partition.k || !feasible) { LOG_ERROR << "*** Partition is infeasible!"; }
}

int main(int argc, char *argv[]) {
  dist::Context ctx;

  { // init MPI
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, ctx.parallel.mpi_thread_support, &provided_thread_support);
    if (provided_thread_support != ctx.parallel.mpi_thread_support) {
      LOG_WARNING << "Desired MPI thread support unavailable: set to " << provided_thread_support;
      if (provided_thread_support == MPI_THREAD_SINGLE) {
        if (dist::mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
          LOG_ERROR << "Your MPI library does not support multithreading. This might cause malfunction.";
        }
        provided_thread_support = MPI_THREAD_FUNNELED; // fake multithreading level for application
      }
      ctx.parallel.mpi_thread_support = provided_thread_support;
    }
  }

  // Parse command line arguments
  try {
    ctx = dist::app::parse_options(argc, argv);
    sanitize_context(ctx);
  } catch (const std::runtime_error &e) { std::cout << e.what() << std::endl; }
  shm::Logger::set_quiet_mode(ctx.quiet);

  shm::print_identifier(argc, argv);
  LOG << "MPI size=" << dist::mpi::get_comm_size(MPI_COMM_WORLD);
  LOG << "CONTEXT " << ctx;

  // Initialize random number generator
  shm::Randomize::seed = ctx.seed;

  // Initialize TBB
  auto gc = shm::init_parallelism(ctx.parallel.num_threads);
  if (ctx.parallel.use_interleaved_numa_allocation) { shm::init_numa(); }

  // Load graph
  const auto graph = TIMED_SCOPE("IO") {
    auto graph = dist::io::metis::read_node_balanced(ctx.graph_filename);
    dist::mpi::barrier(MPI_COMM_WORLD);
    return graph;
  };
  ASSERT([&] { dist::graph::debug::validate(graph); });
  LOG << "Loaded graph with n=" << graph.global_n() << " m=" << graph.global_m();

  // Perform partitioning
  const auto p_graph = TIMED_SCOPE("Partitioning") { return dist::partition(graph, ctx); };
  ASSERT([&] { dist::graph::debug::validate_partition(p_graph); });

  // Output statistics
  print_statistics(p_graph, ctx);

  MPI_Finalize();
  return 0;
}