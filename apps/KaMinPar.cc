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

#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

#if __has_include(<numa.h>)
#include <numa.h>
#endif // __has_include(<numa.h>)

#include "kaminpar-common/environment.h"
#include "kaminpar-common/strutils.h"

#include "apps/io/shm_input_validator.h"
#include "apps/io/shm_io.h"

using namespace kaminpar;
using namespace kaminpar::shm;

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
  bool validate = false;

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
      ->required();
  gp_group->add_option("-G,--graph", app.graph_filename, "Input graph in METIS format.")
      ->configurable(false);

  // Application options
  cli.add_option("-s,--seed", app.seed, "Seed for random number generation.")
      ->default_val(app.seed);
  cli.add_flag("-q,--quiet", app.quiet, "Suppress all console output.");
  cli.add_option("-t,--threads", app.num_threads, "Number of threads to be used.")
      ->check(CLI::NonNegativeNumber)
      ->default_val(app.num_threads);
  cli.add_flag("-E,--experiment", app.experiment, "Use an output format that is easier to parse.");
  cli.add_option(
      "--max-timer-depth", app.max_timer_depth, "Set maximum timer depth shown in result summary."
  );
  cli.add_flag_function("-T,--all-timers", [&](auto) {
    app.max_timer_depth = std::numeric_limits<int>::max();
  });
  cli.add_option("-o,--output", app.partition_filename, "Output filename for the graph partition.")
      ->capture_default_str();
  cli.add_flag(
      "--validate",
      app.validate,
      "Validate input parameters before partitioning (currently only "
      "checks the graph format)."
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
    std::cout << Environment::GIT_SHA1 << std::endl;
    std::exit(0);
  }

  // Allocate graph data structures and read graph file
  StaticArray<EdgeID> xadj;
  StaticArray<NodeID> adjncy;
  StaticArray<NodeWeight> vwgt;
  StaticArray<EdgeWeight> adjwgt;

  if (app.validate) {
    shm::io::metis::read<true>(app.graph_filename, xadj, adjncy, vwgt, adjwgt);
    shm::validate_undirected_graph(xadj, adjncy, vwgt, adjwgt);
  } else {
    shm::io::metis::read<false>(app.graph_filename, xadj, adjncy, vwgt, adjwgt);
  }

  const NodeID n = static_cast<NodeID>(xadj.size() - 1);
  std::vector<BlockID> partition(n);

  EdgeID *xadj_ptr = xadj.data();
  NodeID *adjncy_ptr = adjncy.data();
  NodeWeight *vwgt_ptr = !vwgt.empty() ? vwgt.data() : nullptr;
  EdgeWeight *adjwgt_ptr = !adjwgt.empty() ? adjwgt.data() : nullptr;

  // Compute graph partition
  KaMinPar partitioner(app.num_threads, ctx);
  KaMinPar::reseed(app.seed);

  if (app.quiet) {
    partitioner.set_output_level(OutputLevel::QUIET);
  } else if (app.experiment) {
    partitioner.set_output_level(OutputLevel::EXPERIMENT);
  }

  partitioner.context().debug.graph_name = str::extract_basename(app.graph_filename);
  partitioner.set_max_timer_depth(app.max_timer_depth);
  partitioner.take_graph(n, xadj_ptr, adjncy_ptr, vwgt_ptr, adjwgt_ptr);

  partitioner.compute_partition(app.k, partition.data());

  // Save graph partition
  if (!app.partition_filename.empty()) {
    shm::io::partition::write(app.partition_filename, partition);
  }

  return 0;
}
