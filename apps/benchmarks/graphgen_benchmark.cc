/*******************************************************************************
 * @file:   graphgen_benchmark.cc
 * @author: Daniel Seemaier
 * @date:   29.11.21
 * @brief:  Benchmark for in-memory graph generation.
 ******************************************************************************/
// This must come first since it redefines output macros (LOG DBG etc)
// clang-format off
#include "dkaminpar/definitions.h"
// clang-format on

#include <fstream>

#include <mpi.h>

#include "dkaminpar/graphutils/allgather_graph.h"
#include "dkaminpar/io.h"
#include "dkaminpar/timer.h"

#include "kaminpar/definitions.h"
#include "kaminpar/io.h"

#include "common/logger.h"
#include "common/random.h"
#include "common/timer.h"

#include "apps/apps.h"
#include "apps/dkaminpar/arguments.h"
#include "apps/dkaminpar/graphgen.h"

using namespace kaminpar;
using namespace kaminpar::dist;

int main(int argc, char* argv[]) {
    // Initialize MPI
    {
        int provided_thread_support;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_support);
        if (provided_thread_support != MPI_THREAD_FUNNELED) {
            LOG_WARNING << "Desired MPI thread support unavailable: set to " << provided_thread_support;
            if (provided_thread_support == MPI_THREAD_SINGLE) {
                if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
                    LOG_ERROR << "Your MPI library does not support multithreading. This might cause malfunction.";
                }
            }
        }
    }

    // Parse command line arguments
    auto  app = parse_options(argc, argv);
    auto& ctx = app.ctx;

    // Initialize random number generator
    Random::seed = ctx.seed;

    // Initialize TBB
    auto gc = init_parallelism(ctx.parallel.num_threads);
    if (ctx.parallel.use_interleaved_numa_allocation) {
        init_numa();
    }
    GLOBAL_TIMER.enable(TIMER_BENCHMARK);

    // Load graph
    auto graph = TIMED_SCOPE("IO") {
#ifdef KAMINPAR_GRAPHGEN
        if (app.generator.type != dist::graphgen::GeneratorType::NONE) {
            return dist::graphgen::generate(app.generator);
        }
#endif // KAMINPAR_GRAPHGEN
        return dist::io::read_graph(ctx.graph_filename, dist::io::DistributionType::NODE_BALANCED);
    };

    if (ctx.seed == 42) {
        auto shm_graph = dist::graph::allgather(graph);
        shm::io::metis::write("generated.graph", shm_graph);
    }

    // Print statistics
    {
        const auto n_str       = mpi::gather_statistics_str<dist::GlobalNodeID>(graph.n(), MPI_COMM_WORLD);
        const auto m_str       = mpi::gather_statistics_str<dist::GlobalEdgeID>(graph.m(), MPI_COMM_WORLD);
        const auto ghost_n_str = mpi::gather_statistics_str<dist::GlobalNodeID>(graph.ghost_n(), MPI_COMM_WORLD);

        LOG << "GRAPH "
            << "global_n=" << graph.global_n() << " "
            << "global_m=" << graph.global_m() << " "
            << "n=[" << n_str << "] "
            << "m=[" << m_str << "] "
            << "ghost_n=[" << ghost_n_str << "]";
    }
    dist::graph::print_summary(graph);

    KASSERT(dist::graph::debug::validate(graph));
    ctx.setup(graph);

    // Output statistics
    mpi::barrier(MPI_COMM_WORLD);

    STOP_TIMER();

    finalize_distributed_timer(GLOBAL_TIMER);
    if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
        Timer::global().print_machine_readable(std::cout);
    }
    LOG;
    if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
        Timer::global().print_human_readable(std::cout);
    }

    MPI_Finalize();
    return 0;
}
