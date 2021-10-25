/*******************************************************************************
* @file:   dkaminpar.cc
*
* @author: Daniel Seemaier
* @date:   21.09.21
* @brief:  Distributed KaMinPar binary.
******************************************************************************/
// This must come first since it redefines output macros (LOG DBG etc)
// clang-format off
#include "dkaminpar/distributed_definitions.h"
// clang-format on

#include "apps/apps.h"
#include "dkaminpar/algorithm/locking_clustering_contraction.h"
#include "dkaminpar/application/arguments.h"
#include "dkaminpar/coarsening/locking_label_propagation_clustering.h"
#include "dkaminpar/distributed_context.h"
#include "dkaminpar/distributed_io.h"
#include "kaminpar/definitions.h"
#include "kaminpar/utility/logger.h"
#include "kaminpar/utility/random.h"
#include "kaminpar/utility/timer.h"

#include <fstream>
#include <mpi.h>

namespace dist = dkaminpar;
namespace shm = kaminpar;

int main(int argc, char *argv[]) {
  dist::Context ctx = dist::create_default_context();

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

  // keep alive
  const auto sh = shm::init_backward();
  UNUSED(sh); // hide compile warning if backward is non use

  // Parse command line arguments
  try {
    ctx = dist::app::parse_options(argc, argv);
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
  LOG << "Loaded graph with n=" << graph.global_n() << " m=" << graph.global_m();
  ASSERT([&] { dist::graph::debug::validate(graph); });
  ctx.setup(graph);

  // Perform partitioning
  auto clustering = TIMED_SCOPE("Label Propagation")()->dist::LockingLpClustering::AtomicClusterArray { // force copy
    const auto max_cluster_weight = shm::compute_max_cluster_weight(graph.global_n(), graph.total_node_weight(),
                                                                    ctx.initial_partitioning.sequential.partition,
                                                                    ctx.initial_partitioning.sequential.coarsening);

    dist::LockingLpClustering algorithm(graph.n(), graph.total_n(), ctx.coarsening);
    return algorithm.compute_clustering(graph, max_cluster_weight); // create copy
  };

  dist::mpi::barrier();

  const auto [c_graph, c_mapping, m_ctx] = dist::graph::contract_locking_clustering(graph, clustering);
  LOG << "Coarse graph: n=" << c_graph.n() << " global_n=" << c_graph.global_n();

  // Output statistics
  dist::mpi::barrier();

  if (dist::mpi::get_comm_rank(MPI_COMM_WORLD) == 0 && !ctx.quiet) {
    shm::Timer::global().print_machine_readable(std::cout);
  }
  LOG;
  if (dist::mpi::get_comm_rank(MPI_COMM_WORLD) == 0 && !ctx.quiet) {
    shm::Timer::global().print_human_readable(std::cout);
  }
  LOG;

  MPI_Finalize();
  return 0;
}