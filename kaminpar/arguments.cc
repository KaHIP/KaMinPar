/*******************************************************************************
 * @file:   arguments.cc
 * @author: Daniel Seemaier
 * @date:   14.10.2022
 ******************************************************************************/
#include "kaminpar/arguments.h"

#include "context.h"

#include "kaminpar/context_io.h"

namespace kaminpar::shm {
void create_all_options(CLI::App *app, Context &ctx) {
  create_partitioning_options(app, ctx);
  create_coarsening_options(app, ctx);
  create_lp_coarsening_options(app, ctx);
  create_initial_partitioning_options(app, ctx);
  create_initial_refinement_options(app, ctx);
  create_initial_fm_refinement_options(app, ctx);
  create_refinement_options(app, ctx);
  create_lp_refinement_options(app, ctx);
}

CLI::Option_group *create_partitioning_options(CLI::App *app, Context &ctx) {
  auto *partitioning = app->add_option_group("Partitioning");

  partitioning
      ->add_option("-e,--epsilon", ctx.partition.epsilon,
                   "Maximum allowed imbalance.")
      ->check(CLI::NonNegativeNumber)
      ->configurable(false)
      ->capture_default_str();
  partitioning->add_option("-m,--mode", ctx.partition.mode)
      ->transform(
          CLI::CheckedTransformer(get_partitioning_modes()).description(""))
      ->description(R"(Chooses the partitioning scheme:
  - deep: use deep multilevel graph partitioning
  - rb:   use recursive bisection with plain multilevel graph partitioning)")
      ->capture_default_str();

  return partitioning;
}

CLI::Option_group *create_coarsening_options(CLI::App *app, Context &ctx) {
  auto *coarsening = app->add_option_group("Coarsening");

  coarsening
      ->add_option(
          "-C,--contraction-limit", ctx.coarsening.contraction_limit,
          "Coarse size of a block (parameter C in the ESA'21 publication).")
      ->capture_default_str();
  coarsening
      ->add_option("--clustering-algorithm", ctx.coarsening.algorithm,
                   "Clustering algorithm")
      ->transform(
          CLI::CheckedTransformer(get_clustering_algorithms()).description(""))
      ->description(R"(One of the following options:
  - noop: assign each node to its own cluster, effectively disabling coarsening
  - lp:   use parallel label propagation)")
      ->capture_default_str();
  coarsening
      ->add_option("--cluster-weight-limit",
                   ctx.coarsening.cluster_weight_limit)
      ->transform(
          CLI::CheckedTransformer(get_cluster_weight_limits()).description(""))
      ->description(
          R"(During coarsening, the weight of each cluster is limited by a maximum cluster weight.
This weight is determined by multiplying some base value (set by this option) by some multiplier (set by another option).
Ways to compute the base values are:
  - epsilon-block-weight: Limit cluster weights as described in the ESA'21 publication (c_max = eps * min{n' / C, k})
  - static-block-weight:  Set the cluster weight limit relative to the weight of a block (c_max = n / k)
  - one:                  Set the cluster weight limit to 1 (c_max = 1)
  - zero:                 Set the cluster weight limit to 0 (c_max = 0), effectively disabling coarsening)")
      ->capture_default_str();
  coarsening
      ->add_option("--cluster-weight-multiplier",
                   ctx.coarsening.cluster_weight_multiplier,
                   "Multiplicator of the maximum cluster weight base value.")
      ->capture_default_str();
  coarsening
      ->add_option("--coarsening-convergence-threshold",
                   ctx.coarsening.convergence_threshold,
                   "Coarsening converges once the size of the graph shrinks by "
                   "less than this factor.")
      ->capture_default_str();

  return coarsening;
}

CLI::Option_group *create_lp_coarsening_options(CLI::App *app, Context &ctx) {
  auto *lp = app->add_option_group("Coarsening -> Label Propagation");

  lp->add_option("--c-lp-num-iterations", ctx.coarsening.lp.num_iterations,
                 "Maximum number of label propagation iterations")
      ->capture_default_str();
  lp->add_option("--c-lp-active-large-degree-threshold",
                 ctx.coarsening.lp.large_degree_threshold,
                 "Threshold for ignoring nodes with large degree")
      ->capture_default_str();
  lp->add_option("--c-lp-two-hop-threshold",
                 ctx.coarsening.lp.two_hop_clustering_threshold,
                 "Enable two-hop clustering if plain label propagation shrunk "
                 "the graph by less than this factor")
      ->capture_default_str();
  lp->add_option("--c-lp-max-num-neighbors",
                 ctx.coarsening.lp.max_num_neighbors,
                 "Limit the neighborhood to this many nodes")
      ->capture_default_str();

  return lp;
}

CLI::Option_group *create_initial_partitioning_options(CLI::App *app,
                                                       Context &ctx) {
  auto *ip = app->add_option_group("Initial Partitioning");

  ip->add_option("--i-mode", ctx.initial_partitioning.mode)
      ->transform(CLI::CheckedTransformer(get_initial_partitioning_modes())
                      .description(""))
      ->description(R"(Chooses the initial partitioning mode:
  - sequential:     do not diversify initial partitioning by replicating coarse graphs
  - async-parallel: diversify initial partitioning by replicating coarse graphs each branch of the replication tree asynchronously
  - sync-parallel:  same as async-parallel, but process branches synchronously)")
      ->capture_default_str();
  ip->add_option("--i-rep-exp", ctx.initial_partitioning.multiplier_exponent,
                 "(Deprecated)")
      ->capture_default_str();
  ip->add_option("--i-rep-multiplier",
                 ctx.initial_partitioning.repetition_multiplier,
                 "Multiplier for the number of bipartitioning repetitions")
      ->capture_default_str();
  ip->add_option("--i-min-reps", ctx.initial_partitioning.min_num_repetitions)
      ->description(
          R"(Minimum number of adaptive bipartitioning repetitions per bipartitioning algorithm.
Algorithms might perform less repetitions if they are unlikely to find a competitive bipartition.)")
      ->capture_default_str();
  ip->add_option("--i-min-non-adaptive-reps",
                 ctx.initial_partitioning.min_num_non_adaptive_repetitions)
      ->description(
          R"(Minimum number of adaptive bipartitioning repetitions per bipartitioning algorithm,
before an algorithm is excluded if it is unlikely to find a competitive bipartition.)")
      ->capture_default_str();
  ip->add_option("--i-max-reps", ctx.initial_partitioning.max_num_repetitions,
                 "(Deprecated, but must be larger than the minimum number of "
                 "repetitions)")
      ->capture_default_str();
  ip->add_option("--i-num-seed-iterations",
                 ctx.initial_partitioning.num_seed_iterations,
                 "Number of attempts to find good seeds for BFS-based "
                 "bipartitioning algorithms.")
      ->capture_default_str();
  ip->add_flag("--i-use-adaptive-bipartitioner-selection",
               ctx.initial_partitioning.use_adaptive_bipartitioner_selection,
               "Enable adaptive selection of bipartitioning algorithms.");

  return ip;
}

CLI::Option_group *create_initial_refinement_options(CLI::App *app,
                                                     Context &ctx) {
  auto *refinement =
      app->add_option_group("Initial Partitioning -> Refinement");

  refinement
      ->add_option("--i-r-algorithms",
                   ctx.initial_partitioning.refinement.algorithms)
      ->transform(CLI::CheckedTransformer(get_2way_refinement_algorithms())
                      .description(""))
      ->description(
          R"(Algorithm for 2-way refinement during initial bipartitioning:
  - noop: disable 2-way refinement
  - fm:   use sequential 2-way FM)")
      ->capture_default_str();

  return refinement;
}

CLI::Option_group *create_initial_fm_refinement_options(CLI::App *app,
                                                        Context &ctx) {
  auto *fm = app->add_option_group("Initial Partitioning -> Refinement -> FM");

  fm->add_option("--i-r-fm-stopping-rule",
                 ctx.initial_partitioning.refinement.fm.stopping_rule)
      ->transform(
          CLI::CheckedTransformer(get_fm_stopping_rules()).description(""))
      ->description(R"(Stopping rule for 2-way FM:
  - simple:   abort after a constant number of fruitless moves
  - adaptive: abort after it became unlikely to find a cut improvement)")
      ->capture_default_str();
  fm->add_option("--i-r-fm-num-fruitless-moves",
                 ctx.initial_partitioning.refinement.fm.num_fruitless_moves,
                 "Number of fruitless moves before aborting a FM search "
                 "(simple stopping rule).")
      ->capture_default_str();
  fm->add_option("--i-r-fm-alpha", ctx.initial_partitioning.refinement.fm.alpha,
                 "Alpha factor (adaptive stopping rule).")
      ->capture_default_str();
  fm->add_option("--i-r-fm-num-iterations",
                 ctx.initial_partitioning.refinement.fm.num_iterations,
                 "Number of iterations.")
      ->capture_default_str();
  fm->add_option(
        "--i-r-fm-abortion-threshold",
        ctx.initial_partitioning.refinement.fm.improvement_abortion_threshold,
        "Stop FM iterations if the previous iteration improved the edge cut "
        "below this threshold.")
      ->capture_default_str();

  return fm;
}

CLI::Option_group *create_refinement_options(CLI::App *app, Context &ctx) {
  auto *refinement = app->add_option_group("Refinement");

  refinement->add_option("--r-algorithms", ctx.refinement.algorithms)
      ->transform(CLI::CheckedTransformer(get_kway_refinement_algorithms())
                      .description(""))
      ->description(R"(Algorithm for k-way refinement:
  - noop: disable k-way refinement
  - lp:   use parallel label propagation)")
      ->capture_default_str();

  return refinement;
}

CLI::Option_group *create_lp_refinement_options(CLI::App *app, Context &ctx) {
  auto *lp = app->add_option_group("Refinement -> Label Propagation");

  lp->add_option("--r-lp-num-iterations", ctx.refinement.lp.num_iterations,
                 "Maximum number of label propagation iterations")
      ->capture_default_str();
  lp->add_option("--r-lp-active-large-degree-threshold",
                 ctx.refinement.lp.large_degree_threshold,
                 "Threshold for ignoring nodes with large degree")
      ->capture_default_str();
  lp->add_option("--r-lp-max-num-neighbors",
                 ctx.refinement.lp.max_num_neighbors,
                 "Limit the neighborhood to this many nodes")
      ->capture_default_str();

  return lp;
}
} // namespace kaminpar::shm
