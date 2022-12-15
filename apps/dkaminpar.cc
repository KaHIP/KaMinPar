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
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/definitions.h"
#include "dkaminpar/factories.h"
#include "dkaminpar/graphutils/graph_rearrangement.h"
#include "dkaminpar/io.h"
#include "dkaminpar/logger.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/mpi/utils.h"
#include "dkaminpar/partitioning/partitioner.h"
#include "dkaminpar/presets.h"
#include "dkaminpar/timer.h"

#include "common/console_io.h"
#include "common/random.h"

#include "apps/apps.h"
#include "apps/dkaminpar/graphgen.h"
#include "apps/environment.h"
#include "apps/mpi_apps.h"

using namespace kaminpar;
using namespace kaminpar::dist;

namespace {
struct ApplicationContext {
    // General application settings
    bool quiet              = false;
    int  timer_depth        = 3;
    bool parsable_output    = false;
    bool load_edge_balanced = false;

    // Input
    std::string graph_filename  = "";
    std::string graph_generator = "";

    // Output
    std::string output_partition_filename = "";

    // Algorithmic settings
    int     num_repetitions = 1;
    int     time_limit      = 0;
    Context ctx             = create_default_context();
};

void print_result_statistics(const DistributedPartitionedGraph& p_graph, const ApplicationContext& app) {
    const auto edge_cut  = metrics::edge_cut(p_graph);
    const auto imbalance = metrics::imbalance(p_graph);
    const auto feasible  = metrics::is_feasible(p_graph, app.ctx.partition);

    LOG << "RESULT cut=" << edge_cut << " imbalance=" << imbalance << " feasible=" << feasible << " k=" << p_graph.k();

    // Aggregate timers to display min, max, avg and sd across PEs
    // Disabled: this function requires the same timer hierarchy on all PEs;
    // in deep MGP, this is not always the case
    // if (!ctx.quiet) {
    // finalize_distributed_timer(GLOBAL_TIMER);
    //}

    const bool is_root = mpi::get_comm_rank(MPI_COMM_WORLD) == 0;
    if (is_root && !app.quiet && app.parsable_output) {
        std::cout << "TIME ";
        Timer::global().print_machine_readable(std::cout);
    }
    LOG;
    if (is_root && !app.quiet) {
        Timer::global().print_human_readable(std::cout, app.timer_depth);
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

    if (is_root && (p_graph.k() != app.ctx.partition.k || !feasible)) {
        LOG_ERROR << "*** Partition is infeasible!";
    }
}

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

        auto p_graph = factory::create_partitioner(ctx, graph)->partition();
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

ApplicationContext setup_context(CLI::App& cli, int argc, char* argv[]) {
    ApplicationContext app{};
    bool               dump_config  = false;
    bool               show_version = false;

    cli.set_config("-C,--config", "", "Read parameters from a TOML configuration file.", false);
    cli.add_option_function<std::string>(
           "-P,--preset", [&](const std::string preset) { app.ctx = create_context_by_preset_name(preset); }
    )
        ->check(CLI::IsMember(get_preset_names()))
        ->description(R"(Use configuration preset:
  - default:                    default parameters
  - strong:                     use Mt-KaHyPar for initial partitioning and more label propagation iterations
  - ipdps23-submission-default: dDeepPar-Fast configuration used in the IPDPS'23 submission
  - ipdps23-submission-strong:  dDeepPar-Strong configuration used in the IPDPS'23 submission)");

    // Mandatory
    auto* mandatory = cli.add_option_group("Application")->require_option(1);

    // Mandatory -> either dump config ...
    mandatory->add_flag("--dump-config", dump_config)
        ->configurable(false)
        ->description(R"(Print the current configuration and exit.
The output should be stored in a file and can be used by the -C,--config option.)");
    mandatory->add_flag("-v,--version", show_version, "Show version and exit.");

    // Mandatory -> ... or partition a graph
    auto* gp_group = mandatory->add_option_group("Partitioning")->silent();
    gp_group->add_option("-k,--k", app.ctx.partition.k, "Number of blocks in the partition.")
        ->configurable(false)
        ->required();

    // Graph can come from KaGen or from disk
    auto* graph_source = gp_group->add_option_group("Graph source")->require_option(1)->silent();
#ifdef KAMINPAR_GRAPHGEN
    graph_source->add_option("--generator", app.graph_generator, "Generator properties for in-memory partitioning.")
        ->configurable(false);
#endif // KAMINPAR_GRAPHGEN
    graph_source
        ->add_option(
            "-G,--graph", app.graph_filename,
            "Input graph in METIS (file extension *.graph or *.metis) or binary format (file extension *.bgf)."
        )
        ->configurable(false);

    // Application options
    cli.add_option("-s,--seed", app.ctx.seed, "Seed for random number generation.")->default_val(app.ctx.seed);
    cli.add_flag("-q,--quiet", app.quiet, "Suppress all console output.");
    cli.add_option("-t,--threads", app.ctx.parallel.num_threads, "Number of threads to be used.")
        ->check(CLI::NonNegativeNumber)
        ->default_val(app.ctx.parallel.num_threads);
    cli.add_flag(
           "--edge-balanced", app.load_edge_balanced,
           "Load the input graph such that each PE has roughly the same number of edges."
    )
        ->capture_default_str();
    cli.add_option("-R,--repetitions", app.num_repetitions, "Number of partitioning repetitions to perform.")
        ->capture_default_str();
    cli.add_option("--time-limit", app.time_limit, "Time limit in seconds.")->capture_default_str();
    cli.add_flag("-p,--parsable", app.parsable_output, "Use an output format that is easier to parse.");
    cli.add_option("--timer-depth", app.timer_depth, "Maximum timer depth.");
    cli.add_flag_function(
        "-T,--all-timers", [&](auto) { app.timer_depth = std::numeric_limits<int>::max(); }, "Show all timers."
    );
    cli.add_option("-o,--output", app.output_partition_filename, "Output filename for the graph partition.")
        ->capture_default_str();

    // Algorithmic options
    create_all_options(&cli, app.ctx);

    cli.parse(argc, argv);

    if (dump_config) {
        CLI::App dump;
        create_all_options(&dump, app.ctx);
        std::cout << dump.config_to_str(true, true);
        std::exit(1);
    }

    if (show_version) {
        LOG << Environment::GIT_SHA1;
        std::exit(0);
    }

    return app;
}

void print_parsable_summary(const Context& ctx, const DistributedGraph& graph, const bool root) {
    if (root) {
        cio::print_delimiter(std::cout);
    }
    LOG << "MPI size=" << ctx.parallel.num_mpis;
    LLOG << "CONTEXT ";
    if (root) {
        print_compact(ctx, std::cout, "");
    }

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

void print_execution_mode(const Context& ctx) {
    LOG << "Execution mode:               " << ctx.parallel.num_mpis << " MPI process"
        << (ctx.parallel.num_mpis > 1 ? "es" : "") << " a " << ctx.parallel.num_threads << " thread"
        << (ctx.parallel.num_threads > 1 ? "s" : "");
    cio::print_delimiter();
}
} // namespace

int main(int argc, char* argv[]) {
    init_mpi(argc, argv);
    const PEID size = mpi::get_comm_size(MPI_COMM_WORLD);
    const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);

    //
    // Parse command line arguments
    //
    CLI::App           cli("dKaMinPar: (Somewhat) Minimal Distributed Deep Multilevel Graph Partitioning");
    ApplicationContext app;

    try {
        app                       = setup_context(cli, argc, argv);
        app.ctx.parallel.num_mpis = static_cast<std::size_t>(size);
    } catch (CLI::ParseError& e) {
        return cli.exit(e);
    }

    //
    // Disable console output if requested
    //
    Logger::set_quiet_mode(app.quiet);

    //
    // Print build summary
    //
    if (!app.quiet && rank == 0) {
        cio::print_dkaminpar_banner();
        cio::print_build_identifier<NodeID, EdgeID, shm::NodeWeight, shm::EdgeWeight, NodeWeight, EdgeWeight>(
            Environment::GIT_SHA1, Environment::HOSTNAME
        );
        print_execution_mode(app.ctx);
    }

    //
    // Initialize RNG, setup TBB
    //
    Random::seed = app.ctx.seed;

    auto gc = init_parallelism(app.ctx.parallel.num_threads);
    omp_set_num_threads(static_cast<int>(app.ctx.parallel.num_threads));
    app.ctx.initial_partitioning.kaminpar.parallel.num_threads = app.ctx.parallel.num_threads;
    if (app.ctx.parallel.use_interleaved_numa_allocation) {
        init_numa();
    }

    //
    // Load graph
    //
    auto graph = TIMED_SCOPE("IO") {
        // If configured, generate graph in-memory using KaGen ...
        if (!app.graph_generator.empty()) {
            auto graph         = generate(app.graph_generator);
            app.graph_filename = generate_filename(app.graph_generator);
            if (!app.quiet && rank == 0) {
                cio::print_delimiter(std::cout);
            }
            return graph;
        }

        // ... otherwise, load graph from disk
        const auto type = app.load_edge_balanced ? dist::io::DistributionType::EDGE_BALANCED
                                                 : dist::io::DistributionType::NODE_BALANCED;
        return dist::io::read_graph(app.graph_filename, type);
    };
    KASSERT(graph::debug::validate(graph), "input graph failed graph verification", assert::heavy);

    app.ctx.debug.graph_filename = app.graph_filename;
    app.ctx.setup(graph);

    //
    // Print input summary
    //
    if (!app.quiet) {
        print(app.ctx, rank == 0, std::cout);
        if (app.parsable_output) {
            print_parsable_summary(app.ctx, graph, rank == 0);
        }
        if (rank == 0) {
            cio::print_delimiter();
        }
    }

    //
    // Sort and rearrange graph by degree buckets
    //
    START_TIMER("Partitioning");
    graph = graph::rearrange(std::move(graph), app.ctx);
    STOP_TIMER();

    //
    // Partition graph
    //
    auto p_graph = [&] {
        if (app.num_repetitions > 0 || app.time_limit > 0) {
            if (app.num_repetitions > 0) {
                return partition_repeatedly(
                    graph, app.ctx,
                    [num_repetitions = app.num_repetitions](const int repetition) {
                        return repetition == num_repetitions;
                    }
                );
            } else { // time_limit > 0
                Timer time_limit_timer("");
                return partition_repeatedly(graph, app.ctx, [&time_limit_timer, time_limit = app.time_limit](int) {
                    return time_limit_timer.elapsed_seconds() >= time_limit;
                });
            }
        } else {
            SCOPED_TIMER("Partitioning");
            auto p_graph = factory::create_partitioner(app.ctx, graph)->partition();
            if (!app.quiet && rank == 0) {
                cio::print_delimiter();
            }
            return p_graph;
        }
    }();

    KASSERT(
        graph::debug::validate_partition(p_graph), "graph partition verification failed after partitioning",
        assert::heavy
    );

    //
    // Save partition
    //
    if (!app.output_partition_filename.empty()) {
        dist::io::partition::write(app.output_partition_filename, p_graph);
    }

    //
    // Print statistics
    //
    mpi::barrier(MPI_COMM_WORLD);
    STOP_TIMER(); // stop root timer
    print_result_statistics(p_graph, app);
    return MPI_Finalize();
}
