/*******************************************************************************
 * @file:   dkaminpar.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Distributed KaMinPar binary.
 ******************************************************************************/
// clang-format off
#include "common/CLI11.h"
// clang-format on

#include <fstream>

#include <mpi.h>
#include <omp.h>

#include "dkaminpar/arguments.h"
#include "dkaminpar/context.h"
#include "dkaminpar/context_io.h"
#include "dkaminpar/definitions.h"
#include "dkaminpar/graphutils/graph_rearrangement.h"
#include "dkaminpar/io.h"
#include "dkaminpar/logger.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/mpi/utils.h"
#include "dkaminpar/partitioning/partitioning.h"
#include "dkaminpar/presets.h"
#include "dkaminpar/timer.h"

#include "common/console_io.h"
#include "common/random.h"

#include "apps/apps.h"
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

void print_result_statistics(const DistributedPartitionedGraph& p_graph, const Context& ctx) {
    const auto edge_cut  = metrics::edge_cut(p_graph);
    const auto imbalance = metrics::imbalance(p_graph);
    const auto feasible  = metrics::is_feasible(p_graph, ctx.partition);

    LOG << "RESULT cut=" << edge_cut << " imbalance=" << imbalance << " feasible=" << feasible << " k=" << p_graph.k();
    if (!ctx.quiet) {
        finalize_distributed_timer(GLOBAL_TIMER);
    }

    const bool is_root = mpi::get_comm_rank(MPI_COMM_WORLD) == 0;
    if (is_root && !ctx.quiet && ctx.parsable_output) {
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
        mpi::barrier(MPI_COMM_WORLD);

        Timer repetition_timer("");
        START_TIMER("Partitioning", "Repetition " + std::to_string(repetition));

        auto p_graph = partition(graph, ctx);
        mpi::barrier(MPI_COMM_WORLD);
        STOP_TIMER();
        repetition_timer.stop_timer();

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

        if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
            LOG;
            LOG << "REPETITION run=" << repetition << " cut=" << cut << " imbalance=" << imbalance << " time=" << time
                << " feasible=" << feasible;
            cio::print_delimiter();
        }
    } while (!terminator(results.size()));

    return best_partition;
}

std::pair<Context, GeneratorContext> setup_context(CLI::App& app, int argc, char* argv[]) {
    Context          ctx = create_default_context();
    GeneratorContext g_ctx;
    bool             dump_config = false;

    app.set_config("-C,--config", "", "Read parameters from a TOML configuration file.", false);
    app.add_option_function<std::string>(
           "-P,--preset", [&](const std::string preset) { ctx = create_context_by_preset_name(preset); }
    )
        ->check(CLI::IsMember({"default", "fast", "strong"}))
        ->description(R"(Use configuration preset:
  - default:                    default parameters
  - strong:                     use Mt-KaHyPar for initial partitioning and more label propagation iterations
  - ipdps23-submission-default: dDeepPar-Fast configuration used in the IPDPS'23 submission
  - ipdps23-submission-strong:  dDeepPar-Strong configuration used in the IPDPS'23 submission)");

    // Mandatory
    auto* mandatory = app.add_option_group("Application")->require_option(1);

    // Mandatory -> either dump config ...
    mandatory->add_flag("--dump-config", dump_config)
        ->configurable(false)
        ->description(R"(Print the current configuration and exit.
The output should be stored in a file and can be used by the -C,--config option.)");

    // Mandatory -> ... or partition a graph
    auto* gp_group = mandatory->add_option_group("Partitioning")->silent();
    gp_group->add_option("-k,--k", ctx.partition.k, "Number of blocks in the partition.")
        ->configurable(false)
        ->required();

    // Graph can come from KaGen or from disk
    auto* graph_source = gp_group->add_option_group("Graph source")->require_option(1)->silent();
#ifdef KAMINPAR_ENABLE_GRAPHGEN
    create_generator_options(graph_source, g_ctx);
#endif
    graph_source
        ->add_option(
            "-G,--graph", ctx.graph_filename,
            "Input graph in METIS (file extension *.graph or *.metis) or binary format (file extension *.bgf)."
        )
        ->configurable(false);

    // Application options
    app.add_option("-s,--seed", ctx.seed, "Seed for random number generation.")->default_val(ctx.seed);
    app.add_flag("-q,--quiet", ctx.quiet, "Suppress all console output.");
    app.add_option("-t,--threads", ctx.parallel.num_threads, "Number of threads to be used.")
        ->check(CLI::NonNegativeNumber)
        ->default_val(ctx.parallel.num_threads);
    app.add_flag(
           "--edge-balanced", ctx.load_edge_balanced,
           "Load the input graph such that each PE has roughly the same number of edges."
    )
        ->capture_default_str();
    app.add_option("-R,--repetitions", ctx.num_repetitions, "Number of partitioning repetitions to perform.")
        ->capture_default_str();
    app.add_option("-T,--time-limit", ctx.time_limit, "Time limit in seconds.")->capture_default_str();
    app.add_flag("--sort-graph", ctx.sort_graph, "Rearrange graph by degree buckets after loading it.")
        ->capture_default_str();
    app.add_flag("--simulate-singlethreaded", ctx.parallel.simulate_singlethread, "")->capture_default_str();
    app.add_flag("-p,--parsable", ctx.parsable_output, "Use an output format that is easier to parse.");

    // Algorithmic options
    create_all_options(&app, ctx);

    app.parse(argc, argv);

    if (dump_config) {
        CLI::App dump;
        create_all_options(&dump, ctx);
        std::cout << dump.config_to_str(true, true);
        std::exit(1);
    }

    return {ctx, g_ctx};
}

int main(int argc, char* argv[]) {
    init_mpi(argc, argv);

    CLI::App         app("dKaMinPar: (Somewhat) Minimal Distributed Deep Multilevel Graph Partitioning");
    Context          ctx;
    GeneratorContext g_ctx;

    try {
        std::tie(ctx, g_ctx) = setup_context(app, argc, argv);
    } catch (CLI::ParseError& e) {
        return app.exit(e);
    }

    // Disable console output if requested
    Logger::set_quiet_mode(ctx.quiet);

    const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);
    if (rank == 0) {
        cio::print_dkaminpar_banner();
        cio::print_build_identifier<NodeID, EdgeID, shm::NodeWeight, shm::EdgeWeight, NodeWeight, EdgeWeight>(
            Environment::GIT_SHA1, Environment::HOSTNAME
        );
    }

    ctx.parallel.num_mpis = static_cast<std::size_t>(mpi::get_comm_size(MPI_COMM_WORLD));

    if (ctx.parsable_output) {
        LOG << "MPI size=" << ctx.parallel.num_mpis;
        LLOG << "CONTEXT ";
        print_compact(ctx, std::cout, "");
    }

    // Initialize random number generator
    Random::seed = ctx.seed;
    g_ctx.seed   = ctx.seed;

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
        if (g_ctx.type != GeneratorType::NONE) {
            auto graph         = generate(g_ctx);
            ctx.graph_filename = generate_filename(g_ctx);

            if (g_ctx.save_graph) {
                dist::io::metis::write(ctx.graph_filename, graph, false, false);
            }
            return graph;
        }

        const auto type = ctx.load_edge_balanced ? dist::io::DistributionType::EDGE_BALANCED
                                                 : dist::io::DistributionType::NODE_BALANCED;
        return dist::io::read_graph(ctx.graph_filename, type);
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
    }

    if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
        cio::print_delimiter();
    }

    KASSERT(graph::debug::validate(graph), "", assert::heavy);
    ctx.setup(graph);

    // Sort graph by degree buckets
    if (ctx.sort_graph) {
        SCOPED_TIMER("Partitioning");
        graph = graph::sort_by_degree_buckets(std::move(graph));
        KASSERT(graph::debug::validate(graph), "", assert::heavy);
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
            auto p_graph = partition(graph, ctx);
            if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
                cio::print_delimiter();
            }
            return p_graph;
        }
    }();
    KASSERT(graph::debug::validate_partition(p_graph), "", assert::heavy);

    mpi::barrier(MPI_COMM_WORLD);
    STOP_TIMER(); // stop root timer
    print_result_statistics(p_graph, ctx);

    MPI_Finalize();
    return 0;
}
