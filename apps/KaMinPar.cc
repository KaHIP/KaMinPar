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
#include <span>

#include <tbb/scalable_allocator.h>

#if __has_include(<numa.h>)
#include <numa.h>
#endif // __has_include(<numa.h>)

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/graphutils/permutator.h"

#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/strutils.h"
#include "kaminpar-common/timer.h"

#include "apps/io/shm_input_validator.h"
#include "apps/io/shm_io.h"
#include "apps/io/shm_metis_parser.h"
#include "apps/io/shm_parhip_parser.h"
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
  io::GraphFileFormat input_graph_file_format = io::GraphFileFormat::METIS;

  bool ignore_edge_weights = false;

  std::string partition_filename = "";
  std::string rearranged_graph_filename = "";
  std::string rearranged_mapping_filename = "";
  std::string block_sizes_filename = "";
  io::GraphFileFormat output_graph_file_format = io::GraphFileFormat::METIS;

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
  cli.add_option("-f,--graph-file-format,--input-graph-file-format", app.input_graph_file_format)
      ->transform(CLI::CheckedTransformer(io::get_graph_file_formats()).description(""))
      ->description(R"(Graph file formats:
  - metis
  - parhip)")
      ->capture_default_str();
  cli.add_flag("--ignore-edge-weights", app.ignore_edge_weights, "Ignore edge weights.");

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
  cli.add_option(
         "--output-rearranged-graph",
         app.rearranged_graph_filename,
         "Output filename for the rearranged graph: rearranged input graph such that the vertices "
         "of each block form a consecutive range. The corresponding mapping can be saved using the "
         "--output-rearranged-graph-mapping option."
  )
      ->capture_default_str();
  cli.add_option(
         "--output-rearranged-graph-mapping",
         app.rearranged_mapping_filename,
         "Output filename for the mapping corresponding to the rearranged input graph (see "
         "--output-rerranged-graph, only works in combination with this option)."
  )
      ->capture_default_str();
  cli.add_option("--output-graph-file-format", app.output_graph_file_format)
      ->transform(CLI::CheckedTransformer(io::get_graph_file_formats()).description(""))
      ->description(R"(Graph file formats:
  - metis
  - parhip)")
      ->capture_default_str();
  cli.add_option(
         "--output-block-sizes",
         app.block_sizes_filename,
         "Output the number of vertices in each block (one line per block)."
  )
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

inline void
output_block_sizes(const ApplicationContext &app, const std::vector<BlockID> &partition) {
  shm::io::partition::write_block_sizes(app.block_sizes_filename, app.k, partition);
}

inline void output_partition(const ApplicationContext &app, const std::vector<BlockID> &partition) {
  shm::io::partition::write(app.partition_filename, partition);
}

inline void
output_rearranged_graph(const ApplicationContext &app, const std::vector<BlockID> &partition) {
  if (!app.rearranged_graph_filename.empty()) {
    Graph graph =
        io::read(app.graph_filename, app.input_graph_file_format, NodeOrdering::NATURAL, false);
    auto &csr_graph = graph.concretize<CSRGraph>();

    auto permutations = shm::graph::compute_node_permutation_by_generic_buckets(
        csr_graph.n(), app.k, [&](const NodeID u) { return partition[u]; }
    );

    if (!app.rearranged_mapping_filename.empty()) {
      shm::io::partition::write(app.rearranged_mapping_filename, permutations.old_to_new);
    }

    StaticArray<EdgeID> tmp_nodes(csr_graph.raw_nodes().size());
    StaticArray<NodeID> tmp_edges(csr_graph.raw_edges().size());
    StaticArray<NodeWeight> tmp_node_weights(csr_graph.raw_node_weights().size());
    StaticArray<EdgeWeight> tmp_edge_weights(csr_graph.raw_edge_weights().size());

    shm::graph::build_permuted_graph(
        csr_graph.raw_nodes(),
        csr_graph.raw_edges(),
        csr_graph.raw_node_weights(),
        csr_graph.raw_edge_weights(),
        permutations,
        tmp_nodes,
        tmp_edges,
        tmp_node_weights,
        tmp_edge_weights
    );

    Graph permuted_graph = {std::make_unique<CSRGraph>(
        std::move(tmp_nodes),
        std::move(tmp_edges),
        std::move(tmp_node_weights),
        std::move(tmp_edge_weights)
    )};

    if (app.output_graph_file_format == io::GraphFileFormat::METIS) {
      shm::io::metis::write(app.rearranged_graph_filename, permuted_graph);
    } else if (app.output_graph_file_format == io::GraphFileFormat::PARHIP) {
      shm::io::parhip::write(app.rearranged_graph_filename, permuted_graph.concretize<CSRGraph>());
    } else {
      LOG_WARNING << "Unsupported output graph format";
    }
  }
}

inline void print_rss(const ApplicationContext &app) {
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
        app.graph_filename, app.input_graph_file_format, ctx.node_ordering, ctx.compression.enabled
    );
  };

  if (app.ignore_edge_weights && !ctx.compression.enabled) {
    auto &csr_graph = graph.concretize<CSRGraph>();
    graph = {std::make_unique<CSRGraph>(
        csr_graph.take_raw_nodes(),
        csr_graph.take_raw_edges(),
        csr_graph.take_raw_node_weights(),
        StaticArray<EdgeWeight>{}
    )};
  } else if (app.ignore_edge_weights) {
    LOG_WARNING << "Ignoring edge weights is currently only supported for uncompressed graphs; "
                   "ignoring option.";
  }

  if (app.validate && !ctx.compression.enabled) {
    shm::validate_undirected_graph(graph);
  } else if (app.validate) {
    LOG_WARNING << "Validating the input graph is currently only supported for uncompressed "
                   "graphs; ignoring option.";
  }

  if (static_cast<std::uint64_t>(graph.m()) >
      static_cast<std::uint64_t>(std::numeric_limits<EdgeWeight>::max())) {
    LOG_WARNING << "The edge weight type is not large enough to store the sum of all edge weights. "
                << "This might cause overflows for very large cuts.";
  }

  RECORD("partition") std::vector<BlockID> partition(graph.n());
  RECORD_LOCAL_DATA_STRUCT(partition, partition.capacity() * sizeof(BlockID));
  STOP_HEAP_PROFILER();

  // Compute partition
  partitioner.set_graph(std::move(graph));
  partitioner.compute_partition(app.k, partition.data());

  // Save graph partition
  if (!app.partition_filename.empty()) {
    output_partition(app, partition);
  }

  if (!app.rearranged_graph_filename.empty()) {
    output_rearranged_graph(app, partition);
  }

  if (!app.block_sizes_filename.empty()) {
    output_block_sizes(app, partition);
  }

  DISABLE_HEAP_PROFILER();

  print_rss(app);

  return 0;
}
