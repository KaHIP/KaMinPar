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
#include <tbb/scalable_allocator.h>

#include "kaminpar-common/environment.h"
#include "kaminpar-common/heap_profiler.h"

#include "apps/io/dist_io.h"
#include "apps/io/dist_metis_parser.h"
#include "apps/io/dist_parhip_parser.h"

using namespace kaminpar;
using namespace kaminpar::dist;

namespace {
struct ApplicationContext {
  bool dump_config = false;
  bool show_version = false;

  int seed = 0;
  int num_threads = 1;

  int max_timer_depth = 3;

  bool heap_profiler_detailed = false;
  int heap_profiler_max_depth = 3;
  bool heap_profiler_print_structs = false;
  float heap_profiler_min_struct_size = 10;

  BlockID k = 0;

  bool quiet = false;
  bool experiment = false;
  bool check_input_graph = false;

  kagen::FileFormat io_format = kagen::FileFormat::EXTENSION;
  kagen::GraphDistribution io_distribution = kagen::GraphDistribution::BALANCE_EDGES;

  std::string graph_filename = "";
  std::string partition_filename = "";

  bool no_huge_pages = false;
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
      ->description(
          R"(Graph input format. By default, guess the file format from the file extension. Explicit options are:
  - metis:  text format used by the Metis family
  - parhip: binary format used by ParHiP (+ extensions))"
      )
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

  cli.add_flag("--no-huge-pages", app.no_huge_pages, "Do not use huge pages via TBBmalloc.");

  // Heap profiler options
  if constexpr (kHeapProfiling) {
    auto *hp_group = cli.add_option_group("Heap Profiler");

    hp_group
        ->add_flag(
            "-H,--hp-print-detailed",
            app.heap_profiler_detailed,
            "Show all levels and data structures in the result summary."
        )
        ->default_val(app.heap_profiler_detailed);
    hp_group
        ->add_option(
            "--hp-max-depth",
            app.heap_profiler_max_depth,
            "Set maximum heap profiler depth shown in the result summary."
        )
        ->default_val(app.heap_profiler_max_depth);
    hp_group
        ->add_option(
            "--hp-print-structs",
            app.heap_profiler_print_structs,
            "Print data structure memory statistics in the result summary."
        )
        ->default_val(app.heap_profiler_print_structs);
    hp_group
        ->add_option(
            "--hp-min-struct-size",
            app.heap_profiler_min_struct_size,
            "Sets the minimum size of a data structure in MB to be included in the result summary."
        )
        ->default_val(app.heap_profiler_min_struct_size)
        ->check(CLI::NonNegativeNumber);
  }

  // Algorithmic options
  create_all_options(&cli, ctx);
}

template <typename Lambda> [[noreturn]] void root_run_and_exit(Lambda &&l) {
  const int rank = mpi::get_comm_rank(MPI_COMM_WORLD);
  if (rank == 0) {
    l();
  }
  std::exit(MPI_Finalize());
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

NodeID load_csr_graph(const ApplicationContext &app, dKaMinPar &partitioner) {
  DistributedGraph graph(std::make_unique<DistributedCSRGraph>(
      io::parhip::csr_read(app.graph_filename, false, MPI_COMM_WORLD)
  ));
  const NodeID n = graph.n();

  partitioner.import_graph(std::move(graph));
  return n;
}

NodeID load_compressed_graph(const ApplicationContext &app, dKaMinPar &partitioner) {
  const auto read_graph = [&] {
    switch (app.io_format) {
    case kagen::FileFormat::METIS:
      return io::metis::compress_read(app.graph_filename, false, MPI_COMM_WORLD);
    case kagen::FileFormat::PARHIP:
      return io::parhip::compressed_read(app.graph_filename, false, MPI_COMM_WORLD);
    default:
      root_run_and_exit([&] {
        LOG_ERROR << "Only graphs stored in files with METIS or ParHIP format can be compressed!";
      });
    }
  };

  DistributedGraph graph(std::make_unique<DistributedCompressedGraph>(read_graph()));
  const NodeID n = graph.n();

  partitioner.import_graph(std::move(graph));
  return n;
}

} // namespace

int main(int argc, char *argv[]) {
  int provided, rank;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  CLI::App cli("dKaMinPar: (Somewhat) Minimal Distributed Deep Multilevel "
               "Graph Partitioner");
  ApplicationContext app;
  Context ctx = create_default_context();
  setup_context(cli, app, ctx);
  CLI11_PARSE(cli, argc, argv);

  if (app.dump_config) {
    root_run_and_exit([&] {
      CLI::App dump;
      create_all_options(&dump, ctx);
      std::cout << dump.config_to_str(true, true);
    });
  }

  if (app.show_version) {
    root_run_and_exit([&] { std::cout << Environment::GIT_SHA1 << std::endl; });
  }

  // If available, use huge pages for large allocations
  scalable_allocation_mode(TBBMALLOC_USE_HUGE_PAGES, !app.no_huge_pages);

  ENABLE_HEAP_PROFILER();

  dKaMinPar partitioner(MPI_COMM_WORLD, app.num_threads, ctx);
  dKaMinPar::reseed(app.seed);

  if (app.quiet) {
    partitioner.set_output_level(OutputLevel::QUIET);
  } else if (app.experiment) {
    partitioner.set_output_level(OutputLevel::EXPERIMENT);
  }

  partitioner.context().debug.graph_filename = app.graph_filename;
  partitioner.set_max_timer_depth(app.max_timer_depth);
  if constexpr (kHeapProfiling) {
    auto &global_heap_profiler = heap_profiler::HeapProfiler::global();
    if (app.heap_profiler_detailed) {
      global_heap_profiler.set_detailed_summary_options();
    } else {
      global_heap_profiler.set_max_depth(app.heap_profiler_max_depth);
      global_heap_profiler.set_print_data_structs(app.heap_profiler_print_structs);
      global_heap_profiler.set_min_data_struct_size(app.heap_profiler_min_struct_size);
    }
  }

  START_HEAP_PROFILER("Input Graph Allocation");
  // Load the graph via KaGen or via our graph compressor.
  const NodeID n = [&] {
    if (ctx.compression.enabled) {
      return load_compressed_graph(app, partitioner);
    } else {
      return load_kagen_graph(app, partitioner);
    }
  }();

  // Allocate memory for the partition
  std::vector<BlockID> partition(n);
  STOP_HEAP_PROFILER();

  // Compute the partition
  partitioner.compute_partition(app.k, partition.data());

  if (!app.partition_filename.empty()) {
    dist::io::partition::write(app.partition_filename, partition);
  }

  DISABLE_HEAP_PROFILER();

  return MPI_Finalize();
}
