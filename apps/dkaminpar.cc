/*******************************************************************************
 * @file:   dkaminpar.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Distributed KaMinPar binary.
 ******************************************************************************/
#include <fstream>

#include <mpi.h>
#include <omp.h>

#include "dkaminpar/context.h"
#include "dkaminpar/definitions.h"
#include "dkaminpar/graphutils/graph_rearrangement.h"
#include "dkaminpar/io.h"
#include "dkaminpar/logger.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/partitioning_scheme/partitioning.h"
#include "dkaminpar/timer.h"

#include "common/console_io.h"
#include "common/random.h"

#include "apps/apps.h"
#include "apps/dkaminpar/arguments.h"
#include "apps/dkaminpar/graphgen.h"

using namespace kaminpar;
using namespace kaminpar::dist;

namespace {
void init_mpi(int& argc, char**& argv) {
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

void sanitize_context(const ApplicationContext& app) {
#ifdef KAMINPAR_ENABLE_GRAPHGEN
    if (app.generator.type == GeneratorType::NONE && app.ctx.graph_filename.empty()) {
        FATAL_ERROR << "Must configure a graph generator or specify an input graph";
    }
    if (app.generator.type != GeneratorType::NONE && !app.ctx.graph_filename.empty()) {
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
        finalize_distributed_timer(GLOBAL_TIMER);
    }

    const bool is_root = mpi::get_comm_rank(MPI_COMM_WORLD) == 0;
    if (is_root && !ctx.quiet) {
        std::cout << "TIME ";
        Timer::global().print_machine_readable(std::cout);
    }
    LOG;
    if (is_root && !ctx.quiet) {
        Timer::global().print_human_readable(std::cout);
    }
    LOG;
    LOG << "-> k=" << p_graph.k();
    LOG << "-> cut=" << edge_cut;
    LOG << "-> imbalance=" << imbalance;
    LOG << "-> feasible=" << feasible;
    if (p_graph.k() <= 512) {
        LOG << "-> block_weights:";
        LOG << logger::TABLE << p_graph.block_weights();
    }

    if (is_root && (p_graph.k() != ctx.partition.k || !feasible)) {
        LOG_ERROR << "*** Partition is infeasible!";
    }
}
} // namespace

template <typename Terminator>
DistributedPartitionedGraph
partition_repeatedly(const DistributedGraph& graph, const Context& ctx, Terminator&& terminator) {
    struct Result {
        Result(const double time, const GlobalEdgeWeight cut, const double imbalance, const bool feasible)
            : time(time),
              cut(cut),
              imbalance(imbalance),
              feasible(feasible) {}

        double           time;
        GlobalEdgeWeight cut;
        double           imbalance;
        bool             feasible;
    };
    std::vector<Result> results;

    // Only keep best partition
    DistributedPartitionedGraph best_partition;
    bool                        best_feasible = false;
    GlobalEdgeWeight            best_cut      = kInvalidGlobalEdgeWeight;

    do {
        const std::size_t repetition = results.size();

        Timer repetition_timer("");
        START_TIMER("Partitioning", "Repetition " + std::to_string(repetition));
        auto p_graph = partition(graph, ctx);
        STOP_TIMER();

        // Gather statistics
        const double           time      = repetition_timer.elapsed_seconds();
        const GlobalEdgeWeight cut       = metrics::edge_cut(p_graph);
        const double           imbalance = metrics::imbalance(p_graph);
        const bool             feasible  = metrics::is_feasible(p_graph, ctx.partition);

        // Only keep the partition if it is the best so far
        if (best_cut == kInvalidGlobalEdgeWeight || (!best_feasible && feasible)
            || (best_feasible == feasible && cut < best_cut)) {
            best_partition = std::move(p_graph);
            best_feasible  = feasible;
            best_cut       = cut;
        }

        results.emplace_back(time, cut, imbalance, feasible);
    } while (!terminator(results.size()));

    return best_partition;
}

int main(int argc, char* argv[]) {
    init_mpi(argc, argv);

    // Parse command line arguments
    auto  app = parse_options(argc, argv);
    auto& ctx = app.ctx;
    sanitize_context(app);
    Logger::set_quiet_mode(ctx.quiet);

    const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);
    if (rank == 0) {
        cio::print_dkaminpar_banner();
        cio::print_build_identifier(Environment::GIT_SHA1, Environment::HOSTNAME);
    }

    ctx.parallel.num_mpis = static_cast<std::size_t>(mpi::get_comm_size(MPI_COMM_WORLD));
    LOG << "MPI size=" << ctx.parallel.num_mpis;
    LOG << "CONTEXT " << ctx;

    // Initialize random number generator
    Random::seed = ctx.seed;
#ifdef KAMINPAR_ENABLE_GRAPHGEN
    app.generator.seed = ctx.seed;
#endif

    // Initialize parallelism
    auto gc = init_parallelism(ctx.parallel.num_threads);
    omp_set_num_threads(static_cast<int>(ctx.parallel.num_threads));
    ctx.initial_partitioning.kaminpar.parallel.num_threads = ctx.parallel.num_threads;
    if (ctx.parallel.use_interleaved_numa_allocation) {
        init_numa();
    }

    if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
        cio::print_delimiter();
    }

    // Load graph
    auto graph = TIMED_SCOPE("IO") {
#ifdef KAMINPAR_ENABLE_GRAPHGEN
        if (app.generator.type != GeneratorType::NONE) {
            auto graph         = generate(app.generator);
            ctx.graph_filename = generate_filename(app.generator);

            if (app.generator.save_graph) {
                io::metis::write(ctx.graph_filename, graph, false, false);
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
        const auto n_str       = mpi::gather_statistics_str<GlobalNodeID>(graph.n(), MPI_COMM_WORLD);
        const auto m_str       = mpi::gather_statistics_str<GlobalEdgeID>(graph.m(), MPI_COMM_WORLD);
        const auto ghost_n_str = mpi::gather_statistics_str<GlobalNodeID>(graph.ghost_n(), MPI_COMM_WORLD);

        LOG << "GRAPH "
            << "global_n=" << graph.global_n() << " "
            << "global_m=" << graph.global_m() << " "
            << "n=[" << n_str << "] "
            << "m=[" << m_str << "] "
            << "ghost_n=[" << ghost_n_str << "]";
        LOG;
    }

    KASSERT(graph::debug::validate(graph), "", assert::heavy);
    ctx.setup(graph);

    // If we load the graph from a file, rearrange it so that nodes are sorted by degree buckets
    if (ctx.sort_graph) {
#ifdef KAMINPAR_ENABLE_GRAPHGEN
        if (app.generator.type == GeneratorType::NONE) {
#else
        {
#endif
            SCOPED_TIMER("Partitioning");
            SCOPED_TIMER("Sort graph");
            graph = graph::sort_by_degree_buckets(std::move(graph));
            KASSERT(graph::debug::validate(graph), "", assert::heavy);
        }
    }

    auto p_graph = [&] {
        if (ctx.num_repetitions > 0 || ctx.time_limit > 0) {
            if (ctx.num_repetitions > 0) {
                return partition_repeatedly(
                    graph, ctx,
                    [num_repetitions = ctx.num_repetitions](const std::size_t repetition) {
                        return repetition == num_repetitions;
                    }
                );
            } else { // time_limit > 0
                Timer time_limit_timer("");
                return partition_repeatedly(graph, ctx, [&time_limit_timer, time_limit = ctx.time_limit](std::size_t) {
                    return time_limit_timer.elapsed_seconds() >= time_limit;
                });
            }
        } else {
            SCOPED_TIMER("Partitioning");
            return partition(graph, ctx);
        }
    }();
    KASSERT(graph::debug::validate_partition(p_graph), "", assert::heavy);

    if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
        cio::print_delimiter();
    }

    // Output statistics
    if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
        cio::print_banner("Statistics");
    }

    mpi::barrier(MPI_COMM_WORLD);
    STOP_TIMER(); // stop root timer
    print_result_statistics(p_graph, ctx);

    MPI_Finalize();
    return 0;
}
