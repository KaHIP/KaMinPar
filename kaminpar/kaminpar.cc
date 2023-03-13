/*******************************************************************************
 * @file:   kaminpar.cc
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 * @brief:  Public symbols of the shared-memory partitioner
 ******************************************************************************/
#include "kaminpar/kaminpar.h"

#include "kaminpar/arguments.h"
#include "kaminpar/context.h"
#include "kaminpar/context_io.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/definitions.h"
#include "kaminpar/graphutils/graph_rearrangement.h"
#include "kaminpar/input_validator.h"
#include "kaminpar/io.h"
#include "kaminpar/metrics.h"
#include "kaminpar/partitioning/partitioning.h"
#include "kaminpar/presets.h"

#include "common/assertion_levels.h"
#include "common/console_io.h"
#include "common/logger.h"
#include "common/random.h"
#include "common/timer.h"

namespace kaminpar::shm {
namespace {
void print_statistics(const Context &ctx, const PartitionedGraph &p_graph,
                      const int max_timer_depth, const bool parseable) {
  const EdgeWeight cut = metrics::edge_cut(p_graph);
  const double imbalance = metrics::imbalance(p_graph);
  const bool feasible = metrics::is_feasible(p_graph, ctx.partition);

  cio::print_delimiter("Result Summary");

  // statistics output that is easy to parse
  if (parseable) {
    LOG << "RESULT cut=" << cut << " imbalance=" << imbalance
        << " feasible=" << feasible << " k=" << p_graph.k();
    std::cout << "TIME ";
    Timer::global().print_machine_readable(std::cout);
  }

  Timer::global().print_human_readable(std::cout, max_timer_depth);
  LOG;
  LOG << "Partition summary:";
  if (p_graph.k() != ctx.partition.k) {
    LOG << logger::RED << "  Number of blocks: " << p_graph.k();
  } else {
    LOG << "  Number of blocks: " << p_graph.k();
  }
  LOG << "  Edge cut:         " << cut;
  LOG << "  Imbalance:        " << imbalance;
  if (feasible) {
    LOG << "  Feasible:         yes";
  } else {
    LOG << logger::RED << "  Feasible:         no";
  }
}

std::string generate_partition_filename(const Context &ctx) {
  std::stringstream filename;
  filename << str::extract_basename(ctx.graph_filename);
  filename << "__t" << ctx.parallel.num_threads;
  filename << "__k" << ctx.partition.k;
  filename << "__eps" << ctx.partition.epsilon;
  filename << "__seed" << ctx.seed;
  filename << ".partition";
  return filename.str();
}

Context setup_context(CLI::App &app, int argc, char *argv[]) {
  Context ctx = create_default_context();
  bool dump_config = false;

  app.set_config("-C,--config", "",
                 "Read parameters from a TOML configuration file.", false);
  app.add_option_function<std::string>("-P,--preset",
                                       [&](const std::string preset) {
                                         if (preset == "default") {
                                           ctx = create_default_context();
                                         } else if (preset == "largek") {
                                           ctx = create_largek_context();
                                         }
                                       })
      ->check(CLI::IsMember({"default", "largek"}))
      ->description(R"(Use a configuration preset:
  - default: default parameters
  - largek:  reduce repetitions during initial partitioning (better performance if k is large))");

  // Mandatory
  auto *mandatory_group =
      app.add_option_group("Application")->require_option(1);

  // Mandatory -> either dump config ...
  mandatory_group->add_flag("--dump-config", dump_config)
      ->configurable(false)
      ->description(R"(Print the current configuration and exit.
The output should be stored in a file and can be used by the -C,--config option)");

  // Mandatory -> ... or partition a graph
  auto *gp_group =
      mandatory_group->add_option_group("Partitioning")->silent(true);
  gp_group
      ->add_option("-G,--graph", ctx.graph_filename,
                   "Input graph in METIS file format.")
      ->configurable(false)
      ->required();
  gp_group
      ->add_option("-k,--k", ctx.partition.k,
                   "Number of blocks in the partition.")
      ->configurable(false)
      ->required();

  // Application options
  app.add_option("-s,--seed", ctx.seed, "Seed for random number generation.")
      ->default_val(ctx.seed);
  app.add_option("-o,--output", ctx.partition_filename,
                 "Name of the partition file.")
      ->configurable(false);
  app.add_option("--output-directory", ctx.partition_directory,
                 "Directory in which the partition file should be placed.")
      ->capture_default_str();
  app.add_flag("--degree-weights", ctx.degree_weights,
               "Use node degrees as node weights.");
  app.add_flag("-q,--quiet", ctx.quiet, "Suppress all console output.");
  app.add_option("-t,--threads", ctx.parallel.num_threads,
                 "Number of threads to be used.")
      ->check(CLI::NonNegativeNumber)
      ->default_val(ctx.parallel.num_threads);
  app.add_flag("-p,--parsable", ctx.parsable_output,
               "Use an output format that is easier to parse.");
  app.add_flag(
      "--unchecked-io", ctx.unchecked_io,
      "Run without format checks of the input graph (in Release mode).");
  app.add_flag("--validate-io", ctx.validate_io,
               "Validate the format of the input graph extensively.");

  // Algorithmic options
  create_all_options(&app, ctx);

  app.parse(argc, argv);

  // Only dump config and exit
  if (dump_config) {
    CLI::App dump;
    create_all_options(&dump, ctx);
    std::cout << dump.config_to_str(true, true);
    std::exit(0);
  }

  if (ctx.partition_filename.empty()) {
    ctx.partition_filename = generate_partition_filename(ctx);
  }

  return ctx;
}
} // namespace
KaMinPar::KaMinPar(const int num_threads, const Context ctx)
    : _num_threads(num_threads), _ctx(ctx),
      _gc(tbb::global_control::max_allowed_parallelism, num_threads) {
  Random::seed = 0;
}

void KaMinPar::set_output_level(const OutputLevel output_level) {
  _output_level = output_level;
}

void KaMinPar::set_max_timer_depth(const int max_timer_depth) {
  _max_timer_depth = max_timer_depth;
}

Context &KaMinPar::context() { return _ctx; }

void KaMinPar::import_graph(const NodeID n, EdgeID *xadj, NodeID *adjncy,
                            NodeWeight *vwgt, EdgeWeight *adjwgt) {
  SCOPED_TIMER("IO");

  const EdgeID m = xadj[n];
  const bool has_node_weights = vwgt != nullptr;
  const bool has_edge_weights = adjwgt != nullptr;

  StaticArray<EdgeID> nodes(n + 1);
  StaticArray<NodeID> edges(m);
  StaticArray<NodeWeight> node_weights(has_node_weights * n);
  StaticArray<EdgeWeight> edge_weights(has_edge_weights * m);

  nodes[n] = xadj[n];
  tbb::parallel_for<NodeID>(0, n, [&](const NodeID u) {
    nodes[u] = xadj[u];
    if (has_node_weights) {
      node_weights[u] = vwgt[u];
    }
  });
  tbb::parallel_for<EdgeID>(0, m, [&](const EdgeID e) {
    edges[e] = adjncy[e];
    if (has_edge_weights) {
      edge_weights[e] = adjwgt[e];
    }
  });

  _graph_ptr = std::make_unique<Graph>(std::move(nodes), std::move(edges),
                                       std::move(node_weights),
                                       std::move(edge_weights), false);
}

NodeID KaMinPar::load_graph(const std::string &filename) {
  SCOPED_TIMER("IO");
  _graph_ptr = std::make_unique<Graph>(
      shm::io::metis::read<false>(filename, false, false));
  return _graph_ptr->n();
}

EdgeWeight KaMinPar::compute_partition(const int seed, const BlockID k,
                                       BlockID *partition) {
  cio::print_kaminpar_banner();
  cio::print_build_identifier();
  cio::print_build_datatypes<NodeID, EdgeID, NodeWeight, EdgeWeight>();
  cio::print_delimiter("Input Summary", '#');

  const double original_epsilon = _ctx.partition.epsilon;
  _ctx.parallel.num_threads = _num_threads;
  _ctx.partition.k = k;

  // Setup graph dependent context parameters
  _ctx.setup(*_graph_ptr);

  // Initialize PRNG and console output
  Random::seed = seed;
  Logger::set_quiet_mode(_output_level == OutputLevel::QUIET);

  if (_output_level >= OutputLevel::APPLICATION) {
    print(_ctx, std::cout);
  }

  START_TIMER("Partitioning");
  if (!_was_rearranged) {
    _graph_ptr = std::make_unique<Graph>(
        graph::rearrange_by_degree_buckets(_ctx, std::move(*_graph_ptr)));
    _was_rearranged = true;
  }

  // Perform actual partitioning
  PartitionedGraph p_graph = partitioning::partition(*_graph_ptr, _ctx);

  // Re-integrate isolated nodes that were cut off during preprocessing
  if (_graph_ptr->permuted()) {
    const NodeID num_isolated_nodes =
        graph::integrate_isolated_nodes(*_graph_ptr, original_epsilon, _ctx);
    p_graph = graph::assign_isolated_nodes(std::move(p_graph),
                                           num_isolated_nodes, _ctx.partition);
  }

  START_TIMER("IO");
  if (_graph_ptr->permuted()) {
    tbb::parallel_for<NodeID>(0, p_graph.n(), [&](const NodeID u) {
      partition[u] = p_graph.block(_graph_ptr->map_original_node(u));
    });
  } else {
    tbb::parallel_for<NodeID>(0, p_graph.n(), [&](const NodeID u) {
      partition[u] = p_graph.block(u);
    });
  }
  STOP_TIMER();

  // Print some statistics
  STOP_TIMER(); // stop root timer

  if (_output_level >= OutputLevel::APPLICATION) {
    print_statistics(_ctx, p_graph, _max_timer_depth,
                     _output_level == OutputLevel::EXPERIMENT);
  }

  return metrics::edge_cut(p_graph);
}
} // namespace kaminpar::shm
