/*******************************************************************************
 * @file:   dkaminpar.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Distributed KaMinPar binary.
 ******************************************************************************/
// clang-format off
#include "common/CLI11.h"
// clang-format on

#include "dkaminpar/dkaminpar.h"

#include <fstream>

#include <mpi.h>
#include <omp.h>

#include "dkaminpar/arguments.h"
#include "dkaminpar/io.h"

#include "apps/apps.h"
#include "apps/environment.h"
#include "apps/mpi_apps.h"

using namespace kaminpar;
using namespace kaminpar::dist;

namespace {
struct ApplicationContext {
    int         seed;
    bool        quiet                     = false;
    bool        experiment                = false;
    bool        load_edge_balanced        = false;
    std::string graph_filename            = "";
    std::string output_partition_filename = "";

    Context ctx = create_default_context();
};

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
  - default-social:             default parameters tuned for social networks
  - strong:                     use Mt-KaHyPar for initial partitioning and more label propagation iterations)");

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
    gp_group
        ->add_option(
            "-G,--graph", app.graph_filename,
            "Input graph in METIS (file extension *.graph or *.metis) or binary format (file extension *.bgf)."
        )
        ->configurable(false);

    // Application options
    cli.add_option("-s,--seed", app.seed, "Seed for random number generation.")->default_val(app.seed);
    cli.add_flag("-q,--quiet", app.quiet, "Suppress all console output.");
    cli.add_option("-t,--threads", app.ctx.parallel.num_threads, "Number of threads to be used.")
        ->check(CLI::NonNegativeNumber)
        ->default_val(app.ctx.parallel.num_threads);
    cli.add_flag(
           "--edge-balanced", app.load_edge_balanced,
           "Load the input graph such that each PE has roughly the same number of edges."
    )
        ->capture_default_str();
    cli.add_flag("-E,--experiment", app.experiment, "Use an output format that is easier to parse.");
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
    // Save partition
    //
    if (!app.output_partition_filename.empty()) {
        dist::io::partition::write(app.output_partition_filename, p_graph);
    }

    return MPI_Finalize();
}
