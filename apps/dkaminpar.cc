/*******************************************************************************
 * @file:   dkaminpar.cc
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Distributed KaMinPar binary.
 ******************************************************************************/
// This must come first since it redefines output macros (LOG DBG etc)
// clang-format off
#include "datastructure/distributed_graph.h"
#include "dkaminpar/definitions.h"
// clang-format on

#include <fstream>

#include <mpi.h>
#include <omp.h>

#include "apps/apps.h"
#include "apps/dkaminpar_arguments.h"
#include "apps/dkaminpar_graphgen.h"
#include "dkaminpar/context.h"
#include "dkaminpar/distributed_io.h"
#include "dkaminpar/graphutils/rearrange_graph.h"
#include "dkaminpar/partitioning_scheme/partitioning.h"
#include "dkaminpar/utils/distributed_timer.h"
#include "dkaminpar/utils/metrics.h"
#include "kaminpar/definitions.h"
#include "kaminpar/utils/console_io.h"
#include "kaminpar/utils/logger.h"
#include "kaminpar/utils/random.h"
#include "kaminpar/utils/timer.h"

using namespace dkaminpar;

namespace {
void sanitize_context(const app::ApplicationContext& app) {
#ifdef KAMINPAR_ENABLE_GRAPHGEN
    if (app.generator.type == graphgen::GeneratorType::NONE && app.ctx.graph_filename.empty()) {
        FATAL_ERROR << "Must configure a graph generator or specify an input graph";
    }
    if (app.generator.type != graphgen::GeneratorType::NONE && !app.ctx.graph_filename.empty()) {
        FATAL_ERROR << "cannot configure a graph generator and specify an input graph";
    }
    if (!app.ctx.graph_filename.empty() && !std::ifstream(app.ctx.graph_filename)) {
        FATAL_ERROR << "input graph specified, but file cannot be read";
    }
#else  // KAMINPAR_ENABLE_GRAPHGEN
    if (!std::ifstream(app.ctx.graph_filename)) {
        FATAL_ERROR << "cannot read input graph";
    }
#endif // KAMINPAR_ENABLE_GRAPHGEN
    if (app.ctx.partition.k < 2) {
        FATAL_ERROR << "k must be at least 2.";
    }
    if (app.ctx.partition.epsilon <= 0) {
        FATAL_ERROR << "Epsilon must be greater than zero.";
    }
}

void print_result_statistics(const DistributedPartitionedGraph& p_graph, const Context& ctx) {
    const auto edge_cut  = metrics::edge_cut(p_graph);
    const auto imbalance = metrics::imbalance(p_graph);
    const auto feasible  = metrics::is_feasible(p_graph, ctx.partition);

    LOG << "RESULT cut=" << edge_cut << " imbalance=" << imbalance << " feasible=" << feasible << " k=" << p_graph.k();
    if (!ctx.quiet) {
        timer::collect_and_annotate_distributed_timer(GLOBAL_TIMER);
    }

    const bool is_root = mpi::get_comm_rank(MPI_COMM_WORLD) == 0;
    if (is_root && !ctx.quiet) {
        std::cout << "TIME ";
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

void partition_once(const DistributedGraph& graph, const dkaminpar::Context& ctx, const int repetition) {
    if (repetition > 0) {
        START_TIMER("Partitioning", "Repetition " + std::to_string(repetition));
    } else {
        START_TIMER("Partitioning");
    }
}

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
    auto  app = app::parse_options(argc, argv);
    auto& ctx = app.ctx;
    sanitize_context(app);
    shm::Logger::set_quiet_mode(ctx.quiet);

    shm::print_identifier(argc, argv);
    LOG << "MPI size=" << mpi::get_comm_size(MPI_COMM_WORLD);
    LOG << "CONTEXT " << ctx;

    // Initialize random number generator
    shm::Randomize::seed = ctx.seed;
#ifdef KAMINPAR_ENABLE_GRAPHGEN
    app.generator.seed = ctx.seed;
#endif

    // Initialize parallelism
    auto gc = shm::init_parallelism(ctx.parallel.num_threads);
    omp_set_num_threads(static_cast<int>(ctx.parallel.num_threads));
    ctx.initial_partitioning.sequential.parallel.num_threads = ctx.parallel.num_threads;
    if (ctx.parallel.use_interleaved_numa_allocation) {
        shm::init_numa();
    }

    // Load graph
    auto graph = TIMED_SCOPE("IO") {
#ifdef KAMINPAR_ENABLE_GRAPHGEN
        if (app.generator.type != graphgen::GeneratorType::NONE) {
            auto graph = graphgen::generate(app.generator);
            if (app.generator.save_graph) {
                io::metis::write("generated.graph", graph, false, false);
            }
            return graph;
        }
#endif
        const auto type =
            ctx.load_edge_balanced ? io::DistributionType::EDGE_BALANCED : io::DistributionType::NODE_BALANCED;
        return io::read_graph(ctx.graph_filename, type);
    };

    // Print statistics
    {
        const auto n_str       = mpi::gather_statistics_str<GlobalNodeID>(graph.n());
        const auto m_str       = mpi::gather_statistics_str<GlobalEdgeID>(graph.m());
        const auto ghost_n_str = mpi::gather_statistics_str<GlobalNodeID>(graph.ghost_n());

        LOG << "GRAPH "
            << "global_n=" << graph.global_n() << " "
            << "global_m=" << graph.global_m() << " "
            << "n=[" << n_str << "] "
            << "m=[" << m_str << "] "
            << "ghost_n=[" << ghost_n_str << "]";
    }

    KASSERT(graph::debug::validate(graph));
    ctx.setup(graph);

    // Perform partitioning
    START_TIMER("Partitioning");
    START_TIMER("Sort graph");
    graph = graph::sort_by_degree_buckets(std::move(graph));
    STOP_TIMER();
    const auto p_graph = partition(graph, ctx);
    KASSERT(graph::debug::validate_partition(p_graph));
    STOP_TIMER();

    // Output statistics
    if (mpi::get_comm_rank() == 0) {
        shm::cio::print_banner("Statistics");
    }

    mpi::barrier();
    STOP_TIMER(); // stop root timer
    print_result_statistics(p_graph, ctx);

    MPI_Finalize();
    return 0;
}
