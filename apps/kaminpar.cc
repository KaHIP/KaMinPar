#include "algorithm/graph_utils.h"
#include "apps.h"
#include "application/arguments_parser.h"
#include "application/arguments.h"
#include "datastructure/graph.h"
#include "definitions.h"
#include "io.h"
#include "parallel.h"
#include "partitioning_scheme/partitioning.h"
#include "utility/console_io.h"
#include "utility/logger.h"
#include "utility/metrics.h"
#include "utility/timer.h"

#include <chrono>
#include <fstream>
#include <iostream>

using namespace kaminpar;
using namespace std::string_literals;

void sanitize_context(const Context &context) {
  ALWAYS_ASSERT(!std::ifstream(context.graph_filename) == false) << "Graph file cannot be read. Ensure that the file exists and is readable.";
  ALWAYS_ASSERT(!context.save_partition || !std::ofstream(context.partition_file()) == false) << "Partition file cannot be written to " << context.partition_file() << ". Ensure that the directory exists and is writable.";
  ALWAYS_ASSERT(context.partition.k >= 2) << "k must be at least 2.";
  if (!math::is_power_of_2(context.partition.k)) { WARNING << "k is not a power of 2."; }
  ALWAYS_ASSERT(context.partition.epsilon >= 0) << "Balance constraint cannot be negative.";
  if (context.partition.epsilon == 0) { WARNING << "Balance constraint is set to zero. Note that this software is not designed to compute perfectly balanced partitions. The computed partition will most likely be infeasible."; }

  // Coarsening
  ALWAYS_ASSERT(context.coarsening.min_shrink_factor >= 0);
  ALWAYS_ASSERT(context.coarsening.contraction_limit >= 2) << "Contraction limit must be at least 2.";
  ALWAYS_ASSERT(context.coarsening.shrink_factor_abortion_threshold > 0) << "Abortion threshold of 0 could cause an endless loop during coarsening.";
  ALWAYS_ASSERT(context.coarsening.adaptive_cluster_weight_multiplier >= 0);

  // Initial Partitioning
  ALWAYS_ASSERT(context.initial_partitioning.max_num_repetitions >= context.initial_partitioning.min_num_repetitions) << "Maximum number of repetitions should be at least as large as the minimum number of repetitions.";

  // Initial Partitioning -> Coarsening
  ALWAYS_ASSERT(context.initial_partitioning.coarsening.contraction_limit >= 2);

  // Initial Partitioning -> Refinement -> FM
  if (context.initial_partitioning.refinement.algorithm == RefinementAlgorithm::TWO_WAY_FM) {
    ALWAYS_ASSERT(context.initial_partitioning.refinement.fm.num_iterations > 0) << "If " << RefinementAlgorithm::TWO_WAY_FM << " is set as initial refinement algorithm, we will always perform at least one iteration. To disable initial refinement, use --i-r-algorithm=" << RefinementAlgorithm::NOOP;
  }

  ALWAYS_ASSERT(context.initial_partitioning.parallelize || !context.initial_partitioning.parallelize_synchronized) << "Synchronized parallelized IP requires parallelized IP.";
  ALWAYS_ASSERT(context.initial_partitioning.parallelize || context.initial_partitioning.multiplier_exponent == 0) << "Sequential IP does not support multiplier exponents.";
}
// clang-format on

void print_statistics(const PartitionedGraph &p_graph, const Context &ctx) {
  const EdgeWeight cut = metrics::edge_cut(p_graph);
  const double imbalance = metrics::imbalance(p_graph);
  const bool feasible = metrics::is_feasible(p_graph, ctx.partition);

  // statistics output that is easy to parse
  Timer::global().print_machine_readable(std::cout);
  LOG << "RESULT cut=" << cut << " imbalance=" << imbalance << " feasible=" << feasible << " k=" << p_graph.k();
  LOG;

  // statistics output that is easy to read
  Timer::global().print_human_readable(std::cout);
  LOG;
  timer::FlatTimer::global().print(std::cout);
  LOG;
  LOG << "-> k=" << p_graph.k();
  LOG << "-> cut=" << cut;
  LOG << "-> imbalance=" << imbalance;
  LOG << "-> feasible=" << feasible;
  if (p_graph.k() <= 512) {
    LOG << "-> block weights:";
    LOG << logger::TABLE << p_graph.block_weights();
  }
  if (p_graph.k() != ctx.partition.k || !feasible) { ERROR << "*** Partition is infeasible!"; }
}

std::string generate_partition_filename(const Context &ctx) {
  std::stringstream filename;
  filename << utility::str::extract_basename(ctx.graph_filename);
  filename << "__t" << ctx.parallel.num_threads;
  filename << "__k" << ctx.partition.k;
  filename << "__eps" << ctx.partition.epsilon;
  filename << "__seed" << ctx.seed;
  filename << ".partition";
  return filename.str();
}

int main(int argc, char *argv[]) {
  print_identifier(argc, argv);

  //
  // Parse command line arguments, sanitize, generate output filenames
  //
  Context ctx;
  try {
    ctx = app::parse_options(argc, argv);
    if (ctx.partition_filename.empty()) { ctx.partition_filename = generate_partition_filename(ctx); }
    sanitize_context(ctx);
  } catch (const std::runtime_error &e) { FATAL_ERROR << e.what(); }
  if (ctx.debug.just_sanitize_args) { std::exit(0); }
  if (ctx.debug.force_clean_build) { force_clean_build(); }
  if (!ctx.show_local_timers) { Timer::global().disable_local(); }

  if (ctx.partition.fast_initial_partitioning) {
    ctx.initial_partitioning.min_num_repetitions = 4;
    ctx.initial_partitioning.min_num_non_adaptive_repetitions = 2;
    ctx.initial_partitioning.max_num_repetitions = 4;
  }
  
  //
  // Initialize
  //
  Randomize::seed = ctx.seed;
  auto gc = init_parallelism(ctx.parallel.num_threads); // must stay alive
  if (ctx.parallel.use_interleaved_numa_allocation) { init_numa(); }

  //
  // Load input graph
  //
  bool remove_isolated_nodes = false;
  const double original_epsilon = ctx.partition.epsilon;

  auto [graph, permutations] = [&] {
    StaticArray<EdgeID> nodes;
    StaticArray<NodeID> edges;
    StaticArray<NodeWeight> node_weights;
    StaticArray<EdgeWeight> edge_weights;

    const io::metis::GraphInfo info = TIMED_SCOPE(TIMER_IO) {
      return io::metis::read(ctx.graph_filename, nodes, edges, node_weights, edge_weights);
    };

    START_TIMER(TIMER_PARTITIONING);
    START_TIMER("Preprocessing");

    // sort nodes by degree bucket and rearrange graph, remove isolated nodes
    remove_isolated_nodes = info.has_isolated_nodes && ctx.partition.remove_isolated_nodes;
    NodePermutations permutations = rearrange_and_remove_isolated_nodes(remove_isolated_nodes, ctx.partition, nodes,
                                                                      edges, node_weights, edge_weights,
                                                                      info.total_node_weight);
    STOP_TIMER();
    STOP_TIMER();

    return std::pair{Graph{std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), true},
                     std::move(permutations)};
  }();

  //
  // Setup graph dependent context parameters
  //
  ctx.setup(graph);

  cio::print_banner("Input parameters");
  LOG << "CONTEXT " << ctx;
  LOG << "INPUT graph=" << ctx.graph_filename << " "
      << "n=" << graph.n() << " "
      << "m=" << graph.m() << " "
      << "k=" << ctx.partition.k << " "
      << "epsilon=" << ctx.partition.epsilon << " ";
  LOG << "==> max_block_weight=" << ctx.partition.max_block_weight(0);

  //
  // Perform actual partitioning
  //
  PartitionedGraph p_graph = partitioning::partition(graph, ctx);

  //
  // Re-add isolated nodes (if they were removed)
  //
  if (remove_isolated_nodes) {
    cio::print_banner("Postprocessing");

    START_TIMER(TIMER_PARTITIONING);
    START_TIMER("Postprocessing");

    const NodeID num_nonisolated_nodes = graph.n(); // this becomes the first isolated node
    graph.raw_nodes().unrestrict();
    graph.raw_node_weights().unrestrict();
    graph.update_total_node_weight();
    const NodeID num_isolated_nodes = graph.n() - num_nonisolated_nodes;

    // note: max block weights should not change
    ctx.setup(graph);
    ctx.partition.epsilon = original_epsilon;
    ctx.partition.setup_max_block_weight();

    LOG << "Add " << num_isolated_nodes << " isolated nodes and revert to epsilon=" << original_epsilon;
    LOG << "==> max_block_weight=" << ctx.partition.max_block_weight(0);
    p_graph = revert_isolated_nodes_removal(std::move(p_graph), num_isolated_nodes, ctx.partition);
    STOP_TIMER();
    STOP_TIMER();
  }

  //
  // Store output partition (if requested)
  //
  if (ctx.save_partition) {
    SCOPED_TIMER(TIMER_IO);
    io::partition::write(ctx.partition_file(), p_graph, permutations.old_to_new);
    LOG << "Wrote partition to: " << ctx.partition_file();
  }

  //
  // Print some statistics
  //
  STOP_TIMER(); // stop root timer

  cio::print_banner("Statistics");
  print_statistics(p_graph, ctx);
  return 0;
}
