/*******************************************************************************
 * @file:   dkaminpar.cc
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Distributed KaMinPar binary.
 ******************************************************************************/
// This must come first since it redefines output macros (LOG DBG etc)
// clang-format off
#include "dkaminpar/definitions.h"
// clang-format on

#include "apps/apps.h"
#include "apps/dkaminpar_arguments.h"
#include "apps/dkaminpar_graphgen.h"

#include <fstream>

#include <mpi.h>
#include <omp.h>

#include "dkaminpar/context.h"
#include "dkaminpar/distributed_io.h"
#include "dkaminpar/graphutils/rearrange_graph.h"
#include "dkaminpar/partitioning_scheme/partitioning.h"
#include "dkaminpar/utils/distributed_timer.h"
#include "dkaminpar/utils/metrics.h"
#include "kaminpar/definitions.h"
#include "kaminpar/utils/logger.h"
#include "kaminpar/utils/random.h"
#include "kaminpar/utils/timer.h"

namespace dist = dkaminpar;
namespace shm  = kaminpar;

namespace {
void sanitize_context(const dist::app::ApplicationContext& app) {
#ifdef KAMINPAR_GRAPHGEN
    ALWAYS_ASSERT(app.generator.type != dist::graphgen::GeneratorType::NONE || !app.ctx.graph_filename.empty())
        << "must configure a graph generator or specify an input graph";
    ALWAYS_ASSERT(app.generator.type == dist::graphgen::GeneratorType::NONE || app.ctx.graph_filename.empty())
        << "cannot configure a graph generator and specify an input graph";
    ALWAYS_ASSERT(app.ctx.graph_filename.empty() || !std::ifstream(app.ctx.graph_filename) == false)
        << "input graph specified, but file cannot be read";
#else  // KAMINPAR_GRAPHGEN
    ALWAYS_ASSERT(!std::ifstream(app.ctx.graph_filename) == false) << "cannot read input graph";
#endif // KAMINPAR_GRAPHGEN

    ALWAYS_ASSERT(app.ctx.partition.k >= 2) << "k must be at least 2.";
    ALWAYS_ASSERT(app.ctx.partition.epsilon > 0) << "Epsilon must be greater than zero.";
}

void print_result_statistics(const dist::DistributedPartitionedGraph& p_graph, const dist::Context& ctx) {
    const auto edge_cut  = dist::metrics::edge_cut(p_graph);
    const auto imbalance = dist::metrics::imbalance(p_graph);
    const auto feasible  = dist::metrics::is_feasible(p_graph, ctx.partition);

    LOG << "RESULT cut=" << edge_cut << " imbalance=" << imbalance << " feasible=" << feasible << " k=" << p_graph.k();
    if (!ctx.quiet) {
        dist::timer::collect_and_annotate_distributed_timer(GLOBAL_TIMER);
    }

    const bool is_root = dist::mpi::get_comm_rank(MPI_COMM_WORLD) == 0;
    if (is_root && !ctx.quiet) {
        shm::Timer::global().print_machine_readable(std::cout);
    }
    LOG;
    if (is_root && !ctx.quiet) {
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

    if (is_root && (p_graph.k() != ctx.partition.k || !feasible)) {
        LOG_ERROR << "*** Partition is infeasible!";
    }
}
} // namespace

int main(int argc, char* argv[]) {
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
    auto  app = dist::app::parse_options(argc, argv);
    auto& ctx = app.ctx;
    sanitize_context(app);
    shm::Logger::set_quiet_mode(ctx.quiet);

    shm::print_identifier(argc, argv);
    LOG << "MPI size=" << dist::mpi::get_comm_size(MPI_COMM_WORLD);
    LOG << "CONTEXT " << ctx;

    // Initialize random number generator
    shm::Randomize::seed = ctx.seed;

    // Initialize TBB
    auto gc = shm::init_parallelism(ctx.parallel.num_threads);
    omp_set_num_threads(static_cast<int>(ctx.parallel.num_threads));
    if (ctx.parallel.use_interleaved_numa_allocation) {
        shm::init_numa();
    }

    // Load graph
    auto graph = TIMED_SCOPE("IO") {
#ifdef KAMINPAR_GRAPHGEN
        if (app.generator.type != dist::graphgen::GeneratorType::NONE) {
            auto graph = dist::graphgen::generate(app.generator, ctx.seed);
            if (app.generator.save_graph) {
                dist::io::metis::write("generated.graph", graph, false, false);
            }
            return graph;
        }
#endif // KAMINPAR_GRAPHGEN
        const auto type = ctx.load_edge_balanced ? dist::io::DistributionType::EDGE_BALANCED
                                                 : dist::io::DistributionType::NODE_BALANCED;
        return dist::io::read_graph(ctx.graph_filename, type);
    };

    // Print statistics
    {
        const auto n_str       = dist::mpi::gather_statistics_str<dist::GlobalNodeID>(graph.n());
        const auto m_str       = dist::mpi::gather_statistics_str<dist::GlobalEdgeID>(graph.m());
        const auto ghost_n_str = dist::mpi::gather_statistics_str<dist::GlobalNodeID>(graph.ghost_n());

        LOG << "GRAPH "
            << "global_n=" << graph.global_n() << " "
            << "global_m=" << graph.global_m() << " "
            << "n=[" << n_str << "] "
            << "m=[" << m_str << "] "
            << "ghost_n=[" << ghost_n_str << "]";
    }

    ASSERT([&] { dist::graph::debug::validate(graph); });
    ctx.setup(graph);

    // Perform partitioning
    START_TIMER("Partitioning");
    START_TIMER("Sort graph");
    graph = dist::graph::sort_by_degree_buckets(std::move(graph));
    STOP_TIMER();
    const auto p_graph = dist::partition(graph, ctx);
    ASSERT([&] { dist::graph::debug::validate_partition(p_graph); });
    STOP_TIMER();

    // Output statistics
    dist::mpi::barrier();
    STOP_TIMER(); // stop root timer
    print_result_statistics(p_graph, ctx);

    MPI_Finalize();
    return 0;
}
