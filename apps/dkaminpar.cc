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

#include "apps.h"
#include "dkaminpar/application/arguments.h"
#include "dkaminpar/distributed_context.h"
#include "dkaminpar/distributed_io.h"
#include "dkaminpar/partitioning_scheme/partitioning.h"
#include "dkaminpar/utility/distributed_metrics.h"
#include "dkaminpar/utility/distributed_timer.h"
#include "kaminpar/definitions.h"
#include "kaminpar/utility/logger.h"
#include "kaminpar/utility/random.h"
#include "kaminpar/utility/timer.h"

#include <fstream>
#include <mpi.h>

namespace dist = dkaminpar;
namespace shm = kaminpar;

namespace {
// clang-format off
void sanitize_context(const dist::Context &ctx) {
  ALWAYS_ASSERT(!std::ifstream(ctx.graph_filename) == false) << "Graph file cannot be read. Ensure that the file exists and is readable: " << ctx.graph_filename;
  ALWAYS_ASSERT(ctx.partition.k >= 2) << "k must be at least 2.";
  ALWAYS_ASSERT(ctx.partition.epsilon > 0) << "Epsilon must be greater than zero.";
}
// clang-format on

void print_result_statistics(const dist::DistributedPartitionedGraph &p_graph, const dist::Context &ctx) {
  const auto edge_cut = dist::metrics::edge_cut(p_graph);
  const auto imbalance = dist::metrics::imbalance(p_graph);
  const auto feasible = dist::metrics::is_feasible(p_graph, ctx.partition);

  LOG << "RESULT cut=" << edge_cut << " imbalance=" << imbalance << " feasible=" << feasible << " k=" << p_graph.k();
  if (!ctx.quiet) {
    dist::timer::finalize_distributed_timer(GLOBAL_TIMER);
  }

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

  if (p_graph.k() != ctx.partition.k || !feasible) {
    LOG_ERROR << "*** Partition is infeasible!";
  }
}
} // namespace

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

  // Initialize Backward-Cpp signal handler
  [[maybe_unused]] const auto sh = shm::init_backward();

  // Parse command line arguments
  auto ctx = dist::app::parse_options(argc, argv);
  sanitize_context(ctx);
  shm::Logger::set_quiet_mode(ctx.quiet);

  shm::print_identifier(argc, argv);
  LOG << "MPI size=" << dist::mpi::get_comm_size(MPI_COMM_WORLD);
  LOG << "CONTEXT " << ctx;

  // Initialize random number generator
  shm::Randomize::seed = ctx.seed;

  // Initialize TBB
  auto gc = shm::init_parallelism(ctx.parallel.num_threads);
  if (ctx.parallel.use_interleaved_numa_allocation) {
    shm::init_numa();
  }

  // Load graph
  const auto graph = TIMED_SCOPE("IO") { return dist::io::metis::read_node_balanced(ctx.graph_filename); };
  ctx.refinement.lp.num_chunks = 1024;

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

  // Perform partitioning
  const auto p_graph = TIMED_SCOPE("Partitioning") { return dist::partition(graph, ctx); };
  ASSERT([&] { dist::graph::debug::validate_partition(p_graph); });

  // Output statistics
  dist::mpi::barrier();
  print_result_statistics(p_graph, ctx);

  MPI_Finalize();
  return 0;
}