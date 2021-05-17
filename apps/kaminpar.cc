#include "algorithm/graph_utils.h"
#include "apps.h"
#include "arguments.h"
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

// clang-format off
Context parse_options(int argc, char *argv[]) {
  using namespace std::chrono;
  Context context = Context::create_default();

  Arguments arguments;
  arguments.group("Mandatory", "", true)
      .argument("k", "Number of blocks.", &context.partition.k, 'k')
      .argument("graph", "Graph to partition.", &context.graph_filename, 'G')
      ;

  arguments.group("Miscellaneous", "m")
      .argument("epsilon", "Maximum allowed imbalance.", &context.partition.epsilon, 'e')
      .argument("threads", "Maximum number of threads to be used.", &context.parallel.num_threads, 't')
      .argument("seed", "Seed for random number generator.", &context.seed, 's')
      .argument("randomize", no_argument, [&context](auto *) { context.seed = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count(); }, "Always use a random seed for random number generator. If set, ignore seed argument.", "", 'R')
      .argument("save-partition", "Save the partition to a file.", &context.save_partition)
      .argument("partition-directory", "[--save-partition] Directory for the partition file.", &context.partition_directory)
      .argument("partition-name", "[--save-partition] Filename for the partition file. If empty, one is generated.", &context.partition_filename)
      .argument("use-interleaved-numa-allocation", "Interleave allocations across NUMA nodes round-robin style.", &context.parallel.use_interleaved_numa_allocation)
      .argument("fast-ip", "Use cheaper initial partitioning if k is larger than this.", &context.partition.fast_initial_partitioning)
      .argument("mode", "Partitioning mode, possible values: {" + partitioning_mode_names() + "}.", &context.partition.mode, partitioning_mode_from_string)
      ;

  arguments.group("Debug", "d")
      .argument("d-just-sanitize-args", "Sanitize command line arguments and exit.", &context.debug.just_sanitize_args)
      .argument("d-show-local-timers", "Show thread-local timers.", &context.show_local_timers)
      .argument("d-force-clean-build", "Abort execution if working directory is dirty.", &context.debug.force_clean_build)
      ;

  arguments.group("Coarsening", "c")
      .argument("c-algorithm", "Coarsening algorithm, possible values: {" + coarsening_algorithm_names() + "}. Note: only interleaved coarsening algorithms are well tested.", &context.coarsening.algorithm, coarsening_algorithm_from_string)
      .argument("c-contraction-limit", "Ideally, we always perform a bisection on a graph of size 2 * C.", &context.coarsening.contraction_limit, 'C')
      .argument("c-use-adaptive-contraction-limit", "Use C' = min{C, n/k} as contraction limit to ensure linear running time.", &context.coarsening.use_adaptive_contraction_limit)
      .argument("c-adaptive-cluster-weight-multiplier", "Multiplier for adaptive component of the maximum cluster weight. Set to 0 to disable adaptive cluster weight.", &context.coarsening.adaptive_cluster_weight_multiplier)
      .argument("c-cluster-weight-factor", "Ensure that maximum cluster weight is at least <maximum block weight>/<arg>. Set to max to disable block weight based cluster weight.", &context.coarsening.block_based_cluster_weight_factor)
      .argument("c-shrink-threshold", "Abort coarsening once a level shrunk by factor less than this.", &context.coarsening.shrink_factor_abortion_threshold)
      .argument("c-randomize-chunk-order", "", &context.coarsening.randomize_chunk_order)
      .argument("c-merge-singleton-clusters", "", &context.coarsening.merge_singleton_clusters)
      ;

  arguments.group("Coarsening -> Label Propagation", "c-lp")
      .argument("c-lp-num-iters", "Number of label propagation iterations.", &context.coarsening.num_iterations)
      .argument("c-lp-large-degree-threshold", "Ignore all nodes with degree higher than this during coarsening.", &context.coarsening.large_degree_threshold)
      ;

  arguments.group("Initial Partitioning -> Coarsening", "i-c")
      .argument("i-c-enable", "Enable multilevel initial bipartitioning.", &context.initial_partitioning.coarsening.enable)
      .argument("i-c-cluster-weight-factor", "Determines the maximum cluster weight used during coarsening: c_max = L_max / <double>.", &context.initial_partitioning.coarsening.block_based_cluster_weight_factor)
      .argument("i-c-contraction-limit", "Abort initial coarsening once the number of nodes falls below this parameter.", &context.initial_partitioning.coarsening.contraction_limit)
      ;

  arguments.group("Initial Partitioning", "i")
      .argument("i-multiplier-exp", "", &context.initial_partitioning.multiplier_exponent)
      .argument("i-rep-multiplier", "Multiplier for the number of attempts at computing an initial bisection.", &context.initial_partitioning.repetition_multiplier)
      .argument("i-min-repetitions", "Minimum number of attempts at computing an initial bisection (per bipartition algorithm). A bipartitioning algorithm might be invoked less than specified if it is unlikely to find the best cut.", &context.initial_partitioning.min_num_repetitions)
      .argument("i-min-non-adaptive-repetitions", "Minimum number of attempts at computing an initial bisection (per bipartition algorithm) before excluding bipartitioning algorithms unlikely to find the best cut.", &context.initial_partitioning.min_num_non_adaptive_repetitions)
      .argument("i-max-repetitions", "Maximum number of attempts at computing an initial bisection (per bipartition algorithm).", &context.initial_partitioning.max_num_repetitions)
      .argument("i-num-seed-iterations", "Number of attempts at finding good seed nodes (BFS-based bipartition algorithms).", &context.initial_partitioning.num_seed_iterations)
      .argument("i-use-adaptive-epsilon", "If set, use adaptive epsilon for max block weights during IP.", &context.initial_partitioning.use_adaptive_epsilon)
      .argument("i-use-adaptive-bipartitioner-selection", "If set, determine which bipartitioning algorithms are unlikely to produce good results and run them less often than other algorithms.", &context.initial_partitioning.use_adaptive_bipartitioner_selection)
      .argument("i-parallelize", "Parallelize initial partitioning. This option does not increase speedup, but might improve partition quality.", &context.initial_partitioning.parallelize)
      .argument("i-parallelize-synchronized", "Parallelize initial partitioning and synchronize steps. This option is generally faster than --i-parallelize when using many threads.", &context.initial_partitioning.parallelize_synchronized, 'S')
      ;

  arguments.group("Initial Partitioning -> Refinement", "i-r")
      .argument("i-r-algorithm", "2-way refinement algorithm to be used, possible values: {"s + refinement_algorithm_names() + "}.", &context.initial_partitioning.refinement.algorithm, refinement_algorithm_from_string)
      ;

  arguments.group("Initial Partitioning -> Refinement -> FM", "i-r-fm")
      .argument("i-r-fm-stopping-rule", "Rule used to determine when to stop the FM algorithm, possible values: {"s + fm_stopping_rule_names() + "}.", &context.initial_partitioning.refinement.fm.stopping_rule, fm_stopping_rule_from_string)
      .argument("i-r-fm-num-fruitless-moves", "[Simple stopping rule] Number of fruitless moves after which search is aborted.", &context.initial_partitioning.refinement.fm.num_fruitless_moves)
      .argument("i-r-fm-alpha", "[Adaptive stopping rule] Alpha.", &context.initial_partitioning.refinement.fm.alpha)
      .argument("i-r-fm-iterations", "Maximum number of iterations.", &context.initial_partitioning.refinement.fm.num_iterations)
      .argument("i-r-abortion-threshold", "No more FM iterations if the improvement in edge cut of the last iteration was below this threshold.", &context.initial_partitioning.refinement.fm.improvement_abortion_threshold)
      ;

  arguments.group("Refinement", "r")
      .argument("r-algorithm", "k-way refinement algorithm to be used, possible values: {"s + refinement_algorithm_names() + "}.", &context.refinement.algorithm, refinement_algorithm_from_string)
      ;

  arguments.group("Refinement -> Label Propagation", "r-lp")
      .argument("r-lp-num-iters", "Number label propagation iterations.", &context.refinement.lp.num_iterations)
      .argument("r-lp-large-degree-threshold", "Ignore all nodes with degree higher than this.", &context.refinement.lp.large_degree_threshold)
      .argument("r-lp-randomize-chunk-order", "", &context.refinement.lp.randomize_chunk_order)
      ;

  arguments.group("Refinement -> Balancer", "r-b")
      .argument("r-b-algorithm", "Balancer algorithm, possible values: {"s + balancing_algorithm_names() + "}.", &context.refinement.balancer.algorithm, balancing_algorithm_from_string)
      .argument("r-b-timepoint", "When do we run the balancer, possible values: {"s + balancing_timepoint_names() + "}.", &context.refinement.balancer.timepoint, balancing_timepoint_from_string)
      ;

  arguments.parse(argc, argv);

  return context;
}

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
    ctx = parse_options(argc, argv);
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
