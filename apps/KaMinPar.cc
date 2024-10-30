/*******************************************************************************
 * Standalone binary for the shared-memory partitioner.
 *
 * @file:   KaMinPar.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
// clang-format off
#include "kaminpar-cli/kaminpar_arguments.h"
#include "kaminpar-shm/kaminpar.h"
// clang-format on

#include <iostream>

#include <tbb/scalable_allocator.h>

#if __has_include(<numa.h>)
#include <numa.h>
#endif // __has_include(<numa.h>)

#include "kaminpar-shm/datastructures/graph.h"

#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/strutils.h"
#include "kaminpar-common/timer.h"

#include "apps/io/shm_input_validator.h"
#include "apps/io/shm_io.h"
#include "apps/version.h"

#if defined(__linux__)
#include <sys/resource.h>
#endif

using namespace kaminpar;
using namespace kaminpar::shm;

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
  bool validate = false;
  bool debug = false;

  std::string graph_filename = "";
  std::string partition_filename = "";
  io::GraphFileFormat graph_file_format = io::GraphFileFormat::METIS;

  bool no_huge_pages = false;

  bool dry_run = false;
};

void setup_context(CLI::App &cli, ApplicationContext &app, Context &ctx) {
  cli.set_config("-C,--config", "", "Read parameters from a TOML configuration file.", false);
  cli.add_option_function<std::string>(
         "-P,--preset",
         [&](const std::string preset) { ctx = create_context_by_preset_name(preset); }
  )
      ->check(CLI::IsMember(get_preset_names()))
      ->description(R"(Use configuration preset:
  - fast:    fastest (especially for small graphs), but lowest quality
  - default: in-between
  - strong:  slower, but higher quality (LP + FM)
  - largek:  tuned for k > 1024-ish)");

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
      ->check(CLI::Range(static_cast<BlockID>(2), std::numeric_limits<BlockID>::max()))
      ->required();
  gp_group->add_option("-G,--graph", app.graph_filename, "Input graph in METIS format.")
      ->check(CLI::ExistingFile)
      ->configurable(false);

  // Application options
  cli.add_option("-s,--seed", app.seed, "Seed for random number generation.")
      ->default_val(app.seed);
  cli.add_flag("-q,--quiet", app.quiet, "Suppress all console output.");
  cli.add_option("-t,--threads", app.num_threads, "Number of threads to be used.")
      ->check(CLI::PositiveNumber)
      ->default_val(app.num_threads);
  cli.add_flag("-E,--experiment", app.experiment, "Use an output format that is easier to parse.");
  cli.add_flag(
      "-D,--debug",
      app.debug,
      "Same as -E, but print additional debug information (that might impose a running time "
      "penalty)."
  );
  cli.add_option(
      "--max-timer-depth", app.max_timer_depth, "Set maximum timer depth shown in result summary."
  );
  cli.add_flag_function("-T,--all-timers", [&](auto) {
    app.max_timer_depth = std::numeric_limits<int>::max();
  });
  cli.add_option("-f,--graph-file-format", app.graph_file_format)
      ->transform(CLI::CheckedTransformer(io::get_graph_file_formats()).description(""))
      ->description(R"(Graph file formats:
  - metis
  - parhip)")
      ->capture_default_str();

  if constexpr (kHeapProfiling) {
    auto *hp_group = cli.add_option_group("Heap Profiler");

    hp_group
        ->add_flag(
            "-H,--hp-print-detailed",
            app.heap_profiler_detailed,
            "Show all levels in the result summary."
        )
        ->capture_default_str();
    hp_group
        ->add_option(
            "--hp-max-depth",
            app.heap_profiler_max_depth,
            "Set maximum heap profiler depth shown in the result summary."
        )
        ->capture_default_str();
    hp_group
        ->add_flag(
            "--hp-print-structs",
            app.heap_profiler_print_structs,
            "Print data structure memory statistics in the result summary."
        )
        ->capture_default_str();
    hp_group
        ->add_option(
            "--hp-min-struct-size",
            app.heap_profiler_min_struct_size,
            "Sets the minimum size of a data structure in MiB to be included in the result summary."
        )
        ->capture_default_str()
        ->check(CLI::NonNegativeNumber);
  }

  cli.add_option("-o,--output", app.partition_filename, "Output filename for the graph partition.")
      ->capture_default_str();
  cli.add_flag(
      "--validate",
      app.validate,
      "Validate input parameters before partitioning (currently only "
      "checks the graph format)."
  );
  cli.add_flag("--no-huge-pages", app.no_huge_pages, "Do not use huge pages via TBBmalloc.");

  cli.add_option(
         "--max-overcommitment-factor",
         heap_profiler::max_overcommitment_factor,
         "Limit memory overcommitment to this factor times the total available system memory."
  )
      ->capture_default_str();
  cli.add_flag(
         "--bruteforce-max-overcommitment-factor",
         heap_profiler::bruteforce_max_overcommitment_factor,
         "If enabled, the maximum overcommitment factor is slowly decreased until memory "
         "overcommitment succeeded."
  )
      ->capture_default_str();

  cli.add_flag(
      "--dry-run",
      app.dry_run,
      "Only check the given command line arguments, but do not partition the graph."
  );

  // Algorithmic options
  create_all_options(&cli, ctx);
}
} // namespace

int main(int argc, char *argv[]) {
#if __has_include(<numa.h>)
  if (numa_available() >= 0) {
    numa_set_interleave_mask(numa_all_nodes_ptr);
  }
#endif // __has_include(<numa.h>)

  CLI::App cli("KaMinPar: (Somewhat) Minimal Deep Multilevel Graph Partitioner");
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
    print_version();
    std::exit(0);
  }

  if (app.dry_run) {
    std::exit(0);
  }

  // If available, use huge pages for large allocations
  scalable_allocation_mode(TBBMALLOC_USE_HUGE_PAGES, !app.no_huge_pages);

  ENABLE_HEAP_PROFILER();

  // Setup the KaMinPar instance
  KaMinPar partitioner(app.num_threads, ctx);
  KaMinPar::reseed(app.seed);

  if (app.quiet) {
    partitioner.set_output_level(OutputLevel::QUIET);
  } else if (app.debug) {
    partitioner.set_output_level(OutputLevel::DEBUG);
  } else if (app.experiment) {
    partitioner.set_output_level(OutputLevel::EXPERIMENT);
  }

  partitioner.context().debug.graph_name = str::extract_basename(app.graph_filename);
  partitioner.set_max_timer_depth(app.max_timer_depth);
  if constexpr (kHeapProfiling) {
    auto &global_heap_profiler = heap_profiler::HeapProfiler::global();

    global_heap_profiler.set_max_depth(app.heap_profiler_max_depth);
    global_heap_profiler.set_print_data_structs(app.heap_profiler_print_structs);
    global_heap_profiler.set_min_data_struct_size(app.heap_profiler_min_struct_size);

    if (app.heap_profiler_detailed) {
      global_heap_profiler.set_experiment_summary_options();
    }
  }

  // Read the input graph and allocate memory for the partition
  START_HEAP_PROFILER("Input Graph Allocation");
  Graph graph = TIMED_SCOPE("Read input graph") {
    return io::read(
        app.graph_filename, app.graph_file_format, ctx.node_ordering, ctx.compression.enabled
    );
  };

  if (app.validate) {
    shm::validate_undirected_graph(graph);
  }

  RECORD("partition") std::vector<BlockID> partition(graph.n());
  RECORD_LOCAL_DATA_STRUCT(partition, partition.capacity() * sizeof(BlockID));
  STOP_HEAP_PROFILER();

  // Compute graph partition
  partitioner.set_graph(std::move(graph));
  partitioner.compute_partition(app.k, partition.data());

  // Save graph partition
  if (!app.partition_filename.empty()) {
    shm::io::partition::write(app.partition_filename, partition);
  }

  DISABLE_HEAP_PROFILER();

  if (!app.quiet) {
    std::cout << "\n";

#if defined(__linux__)
    if (struct rusage usage; getrusage(RUSAGE_SELF, &usage) == 0) {
      std::cout << "Maximum resident set size: " << usage.ru_maxrss << " KiB\n";
    } else {
#else
    {
#endif
      std::cout << "Maximum resident set size: unknown\n";
    }

    std::cout << std::flush;
  }

  return 0;
}
