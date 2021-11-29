/*******************************************************************************
 * @file:   graphgen_benchmark.cc
 *
 * @author: Daniel Seemaier
 * @date:   29.11.21
 * @brief:  Benchmark for in-memory graph generation.
 ******************************************************************************/
// This must come first since it redefines output macros (LOG DBG etc)
// clang-format off
#include "dkaminpar/distributed_definitions.h"
// clang-format on

#include "apps/apps.h"
#include "apps/dkaminpar_arguments.h"
//#ifdef KAMINPAR_GRAPHGEN
#include "apps/dkaminpar_graphgen.h"
//#endif // KAMINPAR_GRAPHGEN

#include "dkaminpar/distributed_io.h"
#include "dkaminpar/graphutils/allgather_graph.h"
#include "dkaminpar/utility/distributed_timer.h"
#include "kaminpar/definitions.h"
#include "kaminpar/io.h"
#include "kaminpar/utility/logger.h"
#include "kaminpar/utility/random.h"
#include "kaminpar/utility/timer.h"

#include <fstream>
#include <mpi.h>

namespace dist = dkaminpar;
namespace shm = kaminpar;

int main(int argc, char *argv[]) {
  // Initialize MPI
  {
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_support);
    if (provided_thread_support != MPI_THREAD_FUNNELED) {
      LOG_WARNING << "Desired MPI thread support unavailable: set to " << provided_thread_support;
      if (provided_thread_support == MPI_THREAD_SINGLE) {
        if (dist::mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
          LOG_ERROR << "Your MPI library does not support multithreading. This might cause malfunction.";
        }
      }
    }
  }

  // Parse command line arguments
  auto app = dist::app::parse_options(argc, argv);
  auto &ctx = app.ctx;

  // Initialize random number generator
  shm::Randomize::seed = ctx.seed;

  // Initialize TBB
  auto gc = shm::init_parallelism(ctx.parallel.num_threads);
  if (ctx.parallel.use_interleaved_numa_allocation) {
    shm::init_numa();
  }
  GLOBAL_TIMER.enable(TIMER_BENCHMARK);

  // Load graph
  auto graph = TIMED_SCOPE("IO") {
#ifdef KAMINPAR_GRAPHGEN
    if (app.generator.type != dist::graphgen::GeneratorType::NONE) {
      return dist::graphgen::generate(app.generator);
    }
#endif // KAMINPAR_GRAPHGEN
    return dist::io::read_node_balanced(ctx.graph_filename);
  };

  //auto shm_graph = dist::graph::allgather(graph);
  //shm::io::metis::write("test.graph", shm_graph);

  // Print statistics
  {
    const auto n_str = dist::mpi::gather_statistics_str<dist::GlobalNodeID>(graph.n());
    const auto m_str = dist::mpi::gather_statistics_str<dist::GlobalEdgeID>(graph.m());
    const auto ghost_n_str = dist::mpi::gather_statistics_str<dist::GlobalNodeID>(graph.ghost_n());

    LOG << "GRAPH "
        << "global_n=" << graph.global_n() << " "
        << "global_m=" << graph.global_m() << " "
        << "n=[" << n_str << "] "
        << "m=[" << m_str << "] "
        << "ghost_n=[" << ghost_n_str << "]";
  }
  dist::graph::print_verbose_stats(graph);

  ASSERT([&] { dist::graph::debug::validate(graph); });
  ctx.setup(graph);

  // Output statistics
  dist::mpi::barrier();

  STOP_TIMER();

  // dist::timer::finalize_distributed_timer(GLOBAL_TIMER);
  if (dist::mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    shm::Timer::global().print_machine_readable(std::cout);
  }
  LOG;
  if (dist::mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
    shm::Timer::global().print_human_readable(std::cout);
  }

  MPI_Finalize();
  return 0;
}