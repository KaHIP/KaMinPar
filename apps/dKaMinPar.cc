/*******************************************************************************
 * Standalone binary for the distributed partitioner.
 *
 * @file:   dkaminpar.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
// clang-format off
#include "kaminpar-cli/dkaminpar_arguments.h"
#include "kaminpar-dist/dkaminpar.h"
// clang-format on

#include <kagen.h>
#include <mpi.h>

#include "kaminpar-common/environment.h"
#include "kaminpar-common/strutils.h"

#include "apps/io/dist_io.h"

using namespace kaminpar;
using namespace kaminpar::dist;

namespace {
struct ApplicationContext {
  bool dump_config = false;
  bool show_version = false;

  int seed = 0;
  int num_threads = 1;

  int max_timer_depth = 3;

  BlockID k = 0;

  bool quiet = false;
  bool experiment = false;
  bool check_input_graph = false;

  kagen::FileFormat io_format = kagen::FileFormat::EXTENSION;
  kagen::GraphDistribution io_distribution = kagen::GraphDistribution::BALANCE_EDGES;

  std::string graph_filename = "";
  std::string partition_filename = "";
};

void setup_context(CLI::App &cli, ApplicationContext &app, Context &ctx) {
  cli.set_config("-C,--config", "", "Read parameters from a TOML configuration file.", false);
  cli.add_option_function<std::string>(
         "-P,--preset",
         [&](const std::string preset) { ctx = create_context_by_preset_name(preset); }
  )
      ->check(CLI::IsMember(get_preset_names()))
      ->description(R"(Use configuration preset:
  - default, fast:    fastest, but lower quality
  - strong:           slower, but higher quality
  - europar23-fast:   dKaMinPar-Fast configuration evaluated in the TR / Euro-Par'23 paper
  - europar23-strong: dKaMinPar-Strong configuration evaluated in the TR / Euro-Par'23 paper; requires MtKaHyPar)"
      );

  // Mandatory
  auto *mandatory = cli.add_option_group("Application")->require_option(1);

  // Mandatory -> either dump config ...
  mandatory->add_flag("--dump-config", app.dump_config)
      ->configurable(false)
      ->description(R"(Print the current configuration and exit.
The output should be stored in a file and can be used by the -C,--config option.)");
  mandatory->add_flag("-v,--version", app.show_version, "Show version and exit.");

  // Mandatory -> ... or partition a graph
  auto *gp_group = mandatory->add_option_group("Partitioning")->silent();
  gp_group->add_option("-k,--k", app.k, "Number of blocks in the partition.")
      ->configurable(false)
      ->required();
  gp_group
      ->add_option(
          "-G,--graph",
          app.graph_filename,
          "Input graph in METIS (file extension *.graph or *.metis) "
          "or binary format (file extension *.bgf)."
      )
      ->configurable(false);

  // Application options
  cli.add_option("-s,--seed", app.seed, "Seed for random number generation.")
      ->default_val(app.seed);
  cli.add_flag("-q,--quiet", app.quiet, "Suppress all console output.");
  cli.add_option("-t,--threads", app.num_threads, "Number of threads to be used.")
      ->check(CLI::NonNegativeNumber)
      ->default_val(app.num_threads);
  cli.add_option("--io-format", app.io_format)
      ->transform(CLI::CheckedTransformer(kagen::GetInputFormatMap()).description(""))
      ->description(R"(Graph input format. By default, guess the file format from the file extension. Explicit options are:
  - metis:  text format used by the Metis family
  - parhip: binary format used by ParHiP (+ extensions))")
      ->capture_default_str();
  cli.add_option("--io-distribution", app.io_distribution)
      ->transform(CLI::CheckedTransformer(kagen::GetGraphDistributionMap()).description(""))
      ->description(R"(Graph distribution scheme, possible options are:
  - balance-vertices: distribute vertices such that each PE has roughly the same number of vertices
  - balance-edges:    distribute edges such that each PE has roughly the same number of edges)")
      ->capture_default_str();
  cli.add_flag("-E,--experiment", app.experiment, "Use an output format that is easier to parse.");
  cli.add_option(
      "--max-timer-depth", app.max_timer_depth, "Set maximum timer depth shown in result summary."
  );
  cli.add_flag_function("-T,--all-timers", [&](auto) {
    app.max_timer_depth = std::numeric_limits<int>::max();
  });
  cli.add_option("-o,--output", app.partition_filename, "Output filename for the graph partition.")
      ->capture_default_str();
  cli.add_flag("--check-input-graph", app.check_input_graph, "Check input graph for errors.");

  // Algorithmic options
  create_all_options(&cli, ctx);
}

NodeID load_kagen_graph(const ApplicationContext &app, dKaMinPar &partitioner) {
  using namespace kagen;

  KaGen generator(MPI_COMM_WORLD);
  generator.UseCSRRepresentation();
  if (app.check_input_graph) {
    generator.EnableUndirectedGraphVerification();
  }
  if (app.experiment) {
    generator.EnableBasicStatistics();
    generator.EnableOutput(true);
  }

  Graph graph = [&] {
    if (std::find(app.graph_filename.begin(), app.graph_filename.end(), ';') !=
        app.graph_filename.end()) {
      return generator.GenerateFromOptionString(app.graph_filename);
    } else {
      return generator.ReadFromFile(app.graph_filename, app.io_format, app.io_distribution);
    }
  }();

  // We use `unsigned long` here since we currently do not have any MPI type definitions for
  // GlobalNodeID
  static_assert(std::is_same_v<GlobalNodeID, unsigned long>);
  std::vector<GlobalNodeID> vtxdist =
      BuildVertexDistribution<unsigned long>(graph, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

  // ... if the data types are not the same, we would need to re-allocate memory for the graph; to
  // this if we ever need it ...
  std::vector<SInt> xadj = graph.TakeXadj<>();
  std::vector<SInt> adjncy = graph.TakeAdjncy<>();
  std::vector<SSInt> vwgt = graph.TakeVertexWeights<>();
  std::vector<SSInt> adjwgt = graph.TakeEdgeWeights<>();

  static_assert(sizeof(SInt) == sizeof(GlobalNodeID));
  static_assert(sizeof(SInt) == sizeof(GlobalEdgeID));
  static_assert(sizeof(SSInt) == sizeof(GlobalNodeWeight));
  static_assert(sizeof(SSInt) == sizeof(GlobalEdgeWeight));

  GlobalEdgeID *xadj_ptr = reinterpret_cast<GlobalNodeID *>(xadj.data());
  GlobalNodeID *adjncy_ptr = reinterpret_cast<GlobalNodeID *>(adjncy.data());
  GlobalNodeWeight *vwgt_ptr =
      vwgt.empty() ? nullptr : reinterpret_cast<GlobalNodeWeight *>(vwgt.data());
  GlobalEdgeWeight *adjwgt_ptr =
      adjwgt.empty() ? nullptr : reinterpret_cast<GlobalEdgeWeight *>(adjwgt.data());

  // Pass the graph to the partitioner --
  partitioner.import_graph(vtxdist.data(), xadj_ptr, adjncy_ptr, vwgt_ptr, adjwgt_ptr);

  return graph.vertex_range.second - graph.vertex_range.first;
}
} // namespace

int main(int argc, char *argv[]) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

  CLI::App cli("dKaMinPar: (Somewhat) Minimal Distributed Deep Multilevel "
               "Graph Partitioner");
  ApplicationContext app;
  Context ctx = create_default_context();
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

  dKaMinPar partitioner(MPI_COMM_WORLD, app.num_threads, ctx);
  dKaMinPar::reseed(app.seed);

  if (app.quiet) {
    partitioner.set_output_level(OutputLevel::QUIET);
  } else if (app.experiment) {
    partitioner.set_output_level(OutputLevel::EXPERIMENT);
  }

  partitioner.context().debug.graph_filename = app.graph_filename;
  partitioner.set_max_timer_depth(app.max_timer_depth);

  // Load the graph via KaGen
  const NodeID n = load_kagen_graph(app, partitioner);

  // Compute the partition
  std::vector<BlockID> partition(n);
  partitioner.compute_partition(app.k, partition.data());

  if (!app.partition_filename.empty()) {
    dist::io::partition::write(app.partition_filename, partition);
  }

  return MPI_Finalize();
}
