/*******************************************************************************
 * Command line arguments for the shared-memory partitioner.
 *
 * @file:   kaminpar_arguments.cc
 * @author: Daniel Seemaier
 * @date:   14.10.2022
 ******************************************************************************/
#include "kaminpar_cli/kaminpar_arguments.h"

#include "kaminpar_cli/CLI11.h"

#include "kaminpar/context.h"
#include "kaminpar/context_io.h"

namespace kaminpar::shm {
void create_all_options(CLI::App *app, Context &ctx) {
  create_partitioning_options(app, ctx);
  create_debug_options(app, ctx);
  create_coarsening_options(app, ctx);
  create_initial_partitioning_options(app, ctx);
  create_refinement_options(app, ctx);
}

CLI::Option_group *create_partitioning_options(CLI::App *app, Context &ctx) {
  auto *partitioning = app->add_option_group("Partitioning");

  partitioning
      ->add_option(
          "-e,--epsilon",
          ctx.partition.epsilon,
          "Maximum allowed imbalance, e.g. 0.03 for 3%. Must be strictly "
          "positive."
      )
      ->check(CLI::NonNegativeNumber)
      ->capture_default_str();

  partitioning->add_option("-m,--mode", ctx.mode)
      ->transform(CLI::CheckedTransformer(get_partitioning_modes()).description(""))
      ->description(R"(Partitioning scheme:
  - deep: deep multilevel
  - rb:   recursive multilevel bipartitioning)")
      ->capture_default_str();

  return partitioning;
}

CLI::Option_group *create_coarsening_options(CLI::App *app, Context &ctx) {
  auto *coarsening = app->add_option_group("Coarsening");

  coarsening
      ->add_option(
          "--c-contraction-limit",
          ctx.coarsening.contraction_limit,
          "Upper limit for the number of nodes per block in the coarsest graph."
      )
      ->capture_default_str();

  coarsening->add_option("--c-clustering-algorithm", ctx.coarsening.algorithm)
      ->transform(CLI::CheckedTransformer(get_clustering_algorithms()).description(""))
      ->description(R"(One of the following options:
  - noop: disable coarsening
  - lp:   size-constrained label propagation)")
      ->capture_default_str();

  coarsening->add_option("--c-cluster-weight-limit", ctx.coarsening.cluster_weight_limit)
      ->transform(CLI::CheckedTransformer(get_cluster_weight_limits()).description(""))
      ->description(
          R"(This option selects the formula used to compute the weight limit for nodes in coarse graphs. 
The weight limit can additionally be scaled by a constant multiplier set by the --c-cluster-weight-multiplier option.
Options are:
  - epsilon-block-weight: Cmax = eps * c(V) * min{n' / C, k}, where n' is the number of nodes in the current (coarse) graph
  - static-block-weight:  Cmax = c(V) / k
  - one:                  Cmax = 1
  - zero:                 Cmax = 0 (disable coarsening))"
      )
      ->capture_default_str();

  coarsening
      ->add_option(
          "--c-cluster-weight-multiplier",
          ctx.coarsening.cluster_weight_multiplier,
          "Multiplicator of the maximum cluster weight base value."
      )
      ->capture_default_str();

  coarsening
      ->add_option(
          "--c-coarsening-convergence-threshold",
          ctx.coarsening.convergence_threshold,
          "Coarsening converges once the size of the graph shrinks by "
          "less than this factor."
      )
      ->capture_default_str();

  create_lp_coarsening_options(app, ctx);

  return coarsening;
}

CLI::Option_group *create_lp_coarsening_options(CLI::App *app, Context &ctx) {
  auto *lp = app->add_option_group("Coarsening -> Label Propagation");

  lp->add_option(
        "--c-lp-num-iterations",
        ctx.coarsening.lp.num_iterations,
        "Maximum number of label propagation iterations"
  )
      ->capture_default_str();
  lp->add_option(
        "--c-lp-active-large-degree-threshold",
        ctx.coarsening.lp.large_degree_threshold,
        "Threshold for ignoring nodes with large degree"
  )
      ->capture_default_str();
  lp->add_option(
        "--c-lp-two-hop-threshold",
        ctx.coarsening.lp.two_hop_clustering_threshold,
        "Enable two-hop clustering if plain label propagation shrunk "
        "the graph by less than this factor"
  )
      ->capture_default_str();
  lp->add_option(
        "--c-lp-max-num-neighbors",
        ctx.coarsening.lp.max_num_neighbors,
        "Limit the neighborhood to this many nodes"
  )
      ->capture_default_str();

  return lp;
}

CLI::Option_group *create_initial_partitioning_options(CLI::App *app, Context &ctx) {
  auto *ip = app->add_option_group("Initial Partitioning");

  ip->add_option("--i-mode", ctx.initial_partitioning.mode)
      ->transform(CLI::CheckedTransformer(get_initial_partitioning_modes()).description(""))
      ->description(R"(Chooses the initial partitioning mode:
  - sequential:     do not diversify initial partitioning by replicating coarse graphs
  - async-parallel: diversify initial partitioning by replicating coarse graphs each branch of the replication tree asynchronously
  - sync-parallel:  same as async-parallel, but process branches synchronously)")
      ->capture_default_str();

  /*
  ip->add_option(
        "--i-c-contraction-limit",
        ctx.initial_partitioning.coarsening.contraction_limit,
        "Upper limit for the number of nodes per block in the coarsest graph."
  )
      ->capture_default_str();
  ip->add_option(
        "--i-c-cluster-weight-limit", ctx.initial_partitioning.coarsening.cluster_weight_limit
  )
      ->transform(CLI::CheckedTransformer(get_cluster_weight_limits()).description(""))
      ->description(
          R"(This option selects the formula used to compute the weight limit for nodes in coarse graphs. 
The weight limit can additionally be scaled by a constant multiplier set by the --c-cluster-weight-multiplier option.
Options are:
  - epsilon-block-weight: Cmax = eps * c(V) * min{n' / C, k}, where n' is the number of nodes in the current (coarse) graph
  - static-block-weight:  Cmax = c(V) / k
  - one:                  Cmax = 1
  - zero:                 Cmax = 0 (disable coarsening))"
      )
      ->capture_default_str();
  ip->add_option(
        "--i-c-cluster-weight-multiplier",
        ctx.initial_partitioning.coarsening.cluster_weight_multiplier,
        "Multiplicator of the maximum cluster weight base value."
  )
      ->capture_default_str();
  ip->add_option(
        "--i-c-coarsening-convergence-threshold",
        ctx.initial_partitioning.coarsening.convergence_threshold,
        "Coarsening converges once the size of the graph shrinks by "
        "less than this factor."
  )
      ->capture_default_str();
  */

  /*
  ip->add_option("--i-rep-exp", ctx.initial_partitioning.multiplier_exponent)
      ->capture_default_str();
  ip->add_option("--i-rep-multiplier", ctx.initial_partitioning.repetition_multiplier)
      ->capture_default_str();
  ip->add_option("--i-min-reps", ctx.initial_partitioning.min_num_repetitions)
      ->capture_default_str();
  ip->add_option(
        "--i-min-non-adaptive-reps", ctx.initial_partitioning.min_num_non_adaptive_repetitions
  )
      ->capture_default_str();
  ip->add_option("--i-max-reps", ctx.initial_partitioning.max_num_repetitions)
      ->capture_default_str();
  ip->add_flag(
        "--i-use-adaptive-bipartitioner-selection",
        ctx.initial_partitioning.use_adaptive_bipartitioner_selection
  )
      ->capture_default_str();
  */

  ip->add_flag(
        "--i-r-disable", ctx.initial_partitioning.refinement.disabled, "Disable initial refinement."
  )
      ->capture_default_str();
  
  /*
  ip->add_option("--i-r-stopping-rule", ctx.initial_partitioning.refinement.stopping_rule)
      ->transform(CLI::CheckedTransformer(get_fm_stopping_rules()).description(""))
      ->description(R"(Stopping rule for the 2-way FM algorithm:
  - simple:   abort after a constant number of fruitless moves
  - adaptive: abort after it became unlikely to find a cut improvement)")
      ->capture_default_str();
  ip->add_option(
        "--i-r-num-fruitless-moves",
        ctx.initial_partitioning.refinement.num_fruitless_moves,
        "[--i-r-stopping-rule=simple] Number of fruitless moves before aborting a FM search."
  )
      ->capture_default_str();
  ip->add_option(
        "--i-r-alpha",
        ctx.initial_partitioning.refinement.alpha,
        "[--i-r-stopping-rule=adaptive] Parameter for the adaptive stopping rule."
  )
      ->capture_default_str();
  ip->add_option(
        "--i-r-num-iterations",
        ctx.initial_partitioning.refinement.num_iterations,
        "Number of refinement iterations during initial partitioning."
  )
      ->capture_default_str();
  ip->add_option(
        "--i-r-abortion-threshold",
        ctx.initial_partitioning.refinement.improvement_abortion_threshold,
        "Stop FM iterations if the previous iteration improved the edge cut "
        "below this threshold."
  )
      ->capture_default_str();
  */

  return ip;
}

CLI::Option_group *create_refinement_options(CLI::App *app, Context &ctx) {
  auto *refinement = app->add_option_group("Refinement");

  refinement->add_option("--r-algorithms", ctx.refinement.algorithms)
      ->transform(CLI::CheckedTransformer(get_kway_refinement_algorithms()).description(""))
      ->description(
          R"(This option can be used multiple times to define a sequence of refinement algorithms. 
The following algorithms can be used:
  - noop:            disable k-way refinement
  - lp:              label propagation
  - fm:              FM
  - greedy-balancer: greedy balancer)"
      )
      ->capture_default_str();

  create_lp_refinement_options(app, ctx);
  create_kway_fm_refinement_options(app, ctx);
  create_jet_refinement_options(app, ctx);
  create_mtkahypar_refinement_options(app, ctx);

  return refinement;
}

CLI::Option_group *create_lp_refinement_options(CLI::App *app, Context &ctx) {
  auto *lp = app->add_option_group("Refinement -> Label Propagation");

  lp->add_option(
        "--r-lp-num-iterations",
        ctx.refinement.lp.num_iterations,
        "Number of label propagation iterations to perform"
  )
      ->capture_default_str();

  lp->add_option(
        "--r-lp-active-large-degree-threshold",
        ctx.refinement.lp.large_degree_threshold,
        "Ignore nodes that have a degree larger than this threshold"
  )
      ->capture_default_str();

  lp->add_option(
        "--r-lp-max-num-neighbors",
        ctx.refinement.lp.max_num_neighbors,
        "Maximum number of neighbors to consider for each node"
  )
      ->capture_default_str();

  return lp;
}

CLI::Option_group *create_kway_fm_refinement_options(CLI::App *app, Context &ctx) {
  auto *fm = app->add_option_group("Refinement -> k-way FM");

  fm->add_option(
        "--r-fm-num-iterations",
        ctx.refinement.kway_fm.num_iterations,
        "Number of FM iterations to perform (higher = stronger, but slower)."
  )
      ->capture_default_str();
  fm->add_option(
        "--r-fm-num-seed-nodes",
        ctx.refinement.kway_fm.num_seed_nodes,
        "Number of seed nodes used to initialize a single localized search (lower = stronger, but "
        "slower)."
  )
      ->capture_default_str();
  /*
  fm->add_option("--r-fm-alpha", ctx.refinement.kway_fm.alpha)->capture_default_str();
  */
  /*
  fm->add_flag(
        "--r-fm-use-exact-abortion-threshold", ctx.refinement.kway_fm.use_exact_abortion_threshold
  )
      ->capture_default_str();
  */
  fm->add_option(
        "--r-fm-abortion-threshold",
        ctx.refinement.kway_fm.abortion_threshold,
        "Stop FM iterations if the edge cut reduction of the previous "
        "iteration falls below this threshold (lower = weaker, but faster)."
  )
      ->capture_default_str();
  /*
  fm->add_flag("--r-fm-unlock-seed-nodes", ctx.refinement.kway_fm.unlock_seed_nodes)
      ->capture_default_str();
  */
  fm->add_flag(
        "--r-fm-dbg-batch-size-statistics", ctx.refinement.kway_fm.dbg_compute_batch_size_statistics
  )
      ->capture_default_str();

  return fm;
}

CLI::Option_group *create_jet_refinement_options(CLI::App *app, Context &ctx) {
  auto *jet = app->add_option_group("Refinement -> Jet");

  jet->add_option("--r-jet-num-iterations", ctx.refinement.jet.num_iterations)
      ->capture_default_str();
  /*
  jet->add_flag("--r-jet-interpolate-c", ctx.refinement.jet.interpolate_c)->capture_default_str();
  */
  /*
  jet->add_option("--r-jet-min-c", ctx.refinement.jet.min_c)->capture_default_str();
  */
  /*
  jet->add_option("--r-jet-max-c", ctx.refinement.jet.max_c)->capture_default_str();
  */
  jet->add_option("--r-jet-abortion-threshold", ctx.refinement.jet.abortion_threshold)
      ->capture_default_str();

  return jet;
}

CLI::Option_group *create_mtkahypar_refinement_options(CLI::App *app, Context &ctx) {
  auto *mtkahypar = app->add_option_group("Refinement -> Mt-KaHyPar");

  mtkahypar->add_option("--r-mtkahypar-config-filename", ctx.refinement.mtkahypar.config_filename)
      ->capture_default_str();

  return mtkahypar;
}

CLI::Option_group *create_debug_options(CLI::App *app, Context &ctx) {
  auto *debug = app->add_option_group("Debug");

  debug
      ->add_flag(
          "--d-dump-coarsest-graph",
          ctx.debug.dump_coarsest_graph,
          "Write the coarsest graph to disk. Note that the definition of "
          "'coarsest' depends on the partitioning scheme."
      )
      ->capture_default_str();

  debug
      ->add_flag(
          "--d-dump-coarsest-partition",
          ctx.debug.dump_coarsest_partition,
          "Write partition of the coarsest graph to disk. Note that the "
          "definition of 'coarsest' depends on the partitioning scheme."
      )
      ->capture_default_str();

  debug
      ->add_flag(
          "--d-dump-graph-hierarchy",
          ctx.debug.dump_graph_hierarchy,
          "Write the entire graph hierarchy to disk."
      )
      ->capture_default_str();

  debug
      ->add_flag(
          "--d-dump-partition-hierarchy",
          ctx.debug.dump_partition_hierarchy,
          "Write the entire partition hierarchy to disk."
      )
      ->capture_default_str();

  debug->add_flag(
      "--d-dump-everything",
      [&](auto) {
        ctx.debug.dump_coarsest_graph = true;
        ctx.debug.dump_coarsest_partition = true;
        ctx.debug.dump_graph_hierarchy = true;
        ctx.debug.dump_partition_hierarchy = true;
      },
      "Active all --d-dump-* options."
  );

  return debug;
}
} // namespace kaminpar::shm
