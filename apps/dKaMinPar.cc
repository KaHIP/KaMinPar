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

#include <mpi.h>

#include "dkaminpar/arguments.h"
#include "dkaminpar/io.h"

#include "apps/environment.h"

using namespace kaminpar;
using namespace kaminpar::dist;

namespace {
struct ApplicationContext {
    int         seed                      = 0;
    int         num_threads               = 1;
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    CLI::App           cli("dKaMinPar: (Somewhat) Minimal Distributed Deep Multilevel Graph Partitioning");
    ApplicationContext app;

    try {
        app = setup_context(cli, argc, argv);
    } catch (CLI::ParseError& e) {
        return cli.exit(e);
    }

    // cio::print_build_identifier<NodeID, EdgeID, shm::NodeWeight, shm::EdgeWeight, NodeWeight, EdgeWeight>(
    //     Environment::GIT_SHA1, Environment::HOSTNAME
    //);

    auto graph = dist::io::read_graph(
        app.graph_filename,
        app.load_edge_balanced ? dist::io::DistributionType::EDGE_BALANCED : dist::io::DistributionType::NODE_BALANCED
    );

    app.ctx.debug.graph_filename = app.graph_filename;

    GraphPtr ptr(std::make_unique<DistributedGraph>(std::move(graph)));
    auto     part = compute_graph_partition(std::move(ptr), app.ctx, app.num_threads, app.seed, OutputLevel::FULL);

    if (!app.output_partition_filename.empty()) {
        dist::io::partition::write(app.output_partition_filename, part);
    }

    return MPI_Finalize();
}
