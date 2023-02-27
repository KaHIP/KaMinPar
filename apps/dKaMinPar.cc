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

#include "common/environment.h"

using namespace kaminpar;
using namespace kaminpar::dist;

namespace {
struct ApplicationContext {
    bool dump_config  = false;
    bool show_version = false;

    int seed        = 0;
    int num_threads = 1;

    int max_timer_depth = 3;

    BlockID k = 0;

    bool quiet      = false;
    bool experiment = false;

    bool load_edge_balanced = false;

    std::string graph_filename     = "";
    std::string partition_filename = "";
};

void setup_context(CLI::App& cli, ApplicationContext& app, Context& ctx) {
    cli.set_config("-C,--config", "", "Read parameters from a TOML configuration file.", false);
    cli.add_option_function<std::string>(
           "-P,--preset", [&](const std::string preset) { ctx = create_context_by_preset_name(preset); }
    )
        ->check(CLI::IsMember(get_preset_names()))
        ->description(R"(Use configuration preset:
  - default, fast: default parameters
  - strong:        use Mt-KaHyPar for initial partitioning and more label propagation iterations
  - tr-fast:       dKaMinPar-Fast from the technical report
  - tr-strong:     dKaMinPar-Strong from the technical report)");

    // Mandatory
    auto* mandatory = cli.add_option_group("Application")->require_option(1);

    // Mandatory -> either dump config ...
    mandatory->add_flag("--dump-config", app.dump_config)
        ->configurable(false)
        ->description(R"(Print the current configuration and exit.
The output should be stored in a file and can be used by the -C,--config option.)");
    mandatory->add_flag("-v,--version", app.show_version, "Show version and exit.");

    // Mandatory -> ... or partition a graph
    auto* gp_group = mandatory->add_option_group("Partitioning")->silent();
    gp_group->add_option("-k,--k", app.k, "Number of blocks in the partition.")->configurable(false)->required();
    gp_group
        ->add_option(
            "-G,--graph", app.graph_filename,
            "Input graph in METIS (file extension *.graph or *.metis) or binary format (file extension *.bgf)."
        )
        ->configurable(false);

    // Application options
    cli.add_option("-s,--seed", app.seed, "Seed for random number generation.")->default_val(app.seed);
    cli.add_flag("-q,--quiet", app.quiet, "Suppress all console output.");
    cli.add_option("-t,--threads", app.num_threads, "Number of threads to be used.")
        ->check(CLI::NonNegativeNumber)
        ->default_val(app.num_threads);
    cli.add_flag(
           "--edge-balanced", app.load_edge_balanced,
           "Load the input graph such that each PE has roughly the same number of edges."
    )
        ->capture_default_str();
    cli.add_flag("-E,--experiment", app.experiment, "Use an output format that is easier to parse.");
    cli.add_option("--max-timer-depth", app.max_timer_depth, "Set maximum timer depth shown in result summary.");
    cli.add_flag_function("-T,--all-timers", [&](auto) { app.max_timer_depth = std::numeric_limits<int>::max(); });
    cli.add_option("-o,--output", app.partition_filename, "Output filename for the graph partition.")
        ->capture_default_str();

    // Algorithmic options
    create_all_options(&cli, ctx);
}
} // namespace

int main(int argc, char* argv[]) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    CLI::App           cli("dKaMinPar: (Somewhat) Minimal Distributed Deep Multilevel Graph Partitioning");
    ApplicationContext app;
    Context            ctx = create_default_context();
    setup_context(cli, app, ctx);
    CLI11_PARSE(cli, argc, argv);

    if (app.dump_config) {
        CLI::App dump;
        create_all_options(&dump, ctx);
        std::cout << dump.config_to_str(true, true);
        std::exit(1);
    }

    if (app.show_version) {
        LOG << Environment::GIT_SHA1;
        std::exit(0);
    }

    // cio::print_build_identifier<NodeID, EdgeID, shm::NodeWeight, shm::EdgeWeight, NodeWeight, EdgeWeight>(
    //     Environment::GIT_SHA1, Environment::HOSTNAME
    //);

    DistributedGraphPartitioner partitioner(MPI_COMM_WORLD, app.num_threads, ctx);
    const NodeID n = partitioner.load_graph(app.graph_filename, IOFormat::AUTO, IODistribution::NODE_BALANCED);

    std::vector<BlockID> partition(n);

    partitioner.set_max_timer_depth(app.max_timer_depth);
    partitioner.compute_partition(app.seed, app.k, partition.data());

    if (!app.partition_filename.empty()) {
        dist::io::partition::write(app.partition_filename, partition);
    }

    return MPI_Finalize();
}
