/*******************************************************************************
 * Command line arguments for the distributed partitioner.
 *
 * @file:   dkaminpar_arguments.cc
 * @author: Daniel Seemaier
 * @date:   15.10.2022
 ******************************************************************************/
#include "kaminpar_cli/dkaminpar_arguments.h"

#include "kaminpar_cli/CLI11.h"

#include "dkaminpar/context.h"
#include "dkaminpar/context_io.h"

namespace kaminpar::dist {
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
      ->add_option("-e,--epsilon", ctx.partition.epsilon, "Maximum allowed block imbalance.")
      ->check(CLI::NonNegativeNumber)
      ->configurable(false)
      ->capture_default_str();
  partitioning
      ->add_option(
          "-K,--block-multiplier",
          ctx.partition.K,
          "Maximum block count with which the initial partitioner is called."
      )
      ->capture_default_str();
  partitioning->add_option("-m,--mode", ctx.mode)
      ->transform(CLI::CheckedTransformer(get_partitioning_modes()).description(""))
      ->description(R"(Partitioning scheme, possible options are:
  - deep: distributed deep multilevel graph partitioning
  - deeper: distributed deep multilevel graph partitioning with optional PE splitting and graph replication
  - kway: direct k-way multilevel graph partitioning)")
      ->capture_default_str();
  partitioning
      ->add_flag(
          "--enable-pe-splitting",
          ctx.enable_pe_splitting,
          "Enable PE splitting and graph replication in deep MGP"
      )
      ->capture_default_str();
  partitioning
      ->add_flag(
          "--simulate-singlethreaded",
          ctx.simulate_singlethread,
          "Simulate single-threaded execution during a hybrid run"
      )
      ->capture_default_str();
  partitioning->add_option("--rearrange-by", ctx.rearrange_by)
      ->transform(CLI::CheckedTransformer(get_graph_orderings()).description(""))
      ->description(R"(Criteria by which the graph is sorted and rearrange:
  - natural:     keep order of the graph (do not rearrange)
  - deg-buckets: sort nodes by degree bucket and rearrange accordingly
  - coloring:    color the graph and rearrange accordingly)")
      ->capture_default_str();

  return partitioning;
}

CLI::Option_group *create_debug_options(CLI::App *app, Context &ctx) {
  auto *debug = app->add_option_group("Debug");

  debug->add_flag("--d-save-coarsest-graph", ctx.debug.save_coarsest_graph)
      ->configurable(false)
      ->capture_default_str();
  debug->add_flag("--d-save-coarsest-partition", ctx.debug.save_coarsest_partition)
      ->configurable(false)
      ->capture_default_str();

  return debug;
}

CLI::Option_group *create_initial_partitioning_options(CLI::App *app, Context &ctx) {
  auto *ip = app->add_option_group("Initial Partitioning");

  ip->add_option("--i-algorithm", ctx.initial_partitioning.algorithm)
      ->transform(CLI::CheckedTransformer(get_initial_partitioning_algorithms()).description(""))
      ->description(R"(Algorithm used for initial partitioning. Options are:
  - random:    assign nodes to blocks randomly
  - kaminpar:  use KaMinPar for initial partitioning
  - mtkahypar: use Mt-KaHyPar for inital partitioning)")
      ->capture_default_str();

  return ip;
}

CLI::Option_group *create_refinement_options(CLI::App *app, Context &ctx) {
  auto *refinement = app->add_option_group("Refinement");

  refinement->add_option("--r-algorithm,--r-algorithms", ctx.refinement.algorithms)
      ->transform(CLI::CheckedTransformer(get_kway_refinement_algorithms()).description(""))
      ->description(
          R"(K-way refinement algorithm(s). Possible options are (separated by space):
  - noop:            disable k-way refinement
  - colored-lp:      distributed label propagation based on node coloring
  - lp:              distributed label propagation
  - local-fm:        PE-local FM
  - fm:              distributed FM
  - greedy_balancer: greedy algorithm to force balance)"
      )
      ->capture_default_str();
  refinement
      ->add_flag(
          "--r-refine-coarsest-graph",
          ctx.refinement.refine_coarsest_level,
          "Also run the refinement algorithms on the coarsest graph."
      )
      ->capture_default_str();

  create_fm_refinement_options(app, ctx);
  create_lp_refinement_options(app, ctx);
  create_colored_lp_refinement_options(app, ctx);
  create_jet_refinement_options(app, ctx);
  create_greedy_balancer_options(app, ctx);

  return refinement;
}

CLI::Option_group *create_fm_refinement_options(CLI::App *app, Context &ctx) {
  auto *fm = app->add_option_group("Refinement -> FM");

  fm->add_option(
        "--r-fm-alpha", ctx.refinement.fm.alpha, "Alpha parameter for the adaptive stopping rule."
  )
      ->capture_default_str();
  fm->add_option("--r-fm-radius", ctx.refinement.fm.radius, "Radius for the search graphs.")
      ->capture_default_str();
  fm->add_option(
        "--r-fm-hops", ctx.refinement.fm.pe_radius, "Number of PE hops for the BFS search."
  )
      ->capture_default_str();
  fm->add_flag(
        "--r-fm-overlap-regions",
        ctx.refinement.fm.overlap_regions,
        "Allow search regions to overlap."
  )
      ->capture_default_str();
  fm->add_option("--r-fm-iterations", ctx.refinement.fm.num_iterations, "Number of FM iterations.")
      ->capture_default_str();
  fm->add_flag(
        "--r-fm-sequential", ctx.refinement.fm.sequential, "Refine search graphs sequentially."
  )
      ->capture_default_str();
  fm->add_flag(
        "--r-fm-premove-locally",
        ctx.refinement.fm.premove_locally,
        "Move nodes right away, i.e., before global synchronization steps."
  )
      ->capture_default_str();
  fm->add_option(
        "--r-fm-bound-degree",
        ctx.refinement.fm.bound_degree,
        "Add at most this many neighbors of a high-degree node to the search "
        "region."
  )
      ->capture_default_str();
  fm->add_flag(
        "--r-fm-contract-border",
        ctx.refinement.fm.contract_border,
        "Contract the exterior of the search graph"
  )
      ->capture_default_str();

  return fm;
}

CLI::Option_group *create_lp_refinement_options(CLI::App *app, Context &ctx) {
  auto *lp = app->add_option_group("Refinement -> Chunked Label Propagation");

  lp->add_option(
        "--r-lp-num-iterations",
        ctx.refinement.lp.num_iterations,
        "Number of label propagation iterations."
  )
      ->capture_default_str();
  lp->add_option(
        "--r-lp-total-chunks",
        ctx.refinement.lp.total_num_chunks,
        "Number of synchronization rounds times number of PEs."
  )
      ->capture_default_str();
  lp->add_option(
        "--r-lp-min-chunks",
        ctx.refinement.lp.min_num_chunks,
        "Minimum number of synchronization rounds."
  )
      ->capture_default_str();
  lp->add_option(
        "--r-lp-num-chunks",
        ctx.refinement.lp.fixed_num_chunks,
        "Set the number of chunks to a fixed number rather than deducing it "
        "from other parameters (0 = deduce)."
  )
      ->capture_default_str();
  lp->add_option(
        "--r-lp-active-large-degree-threshold",
        ctx.refinement.lp.active_high_degree_threshold,
        "Do not move nodes with degree larger than this."
  )
      ->capture_default_str();
  lp->add_flag(
        "--r-lp-ignore-probabilities", ctx.refinement.lp.ignore_probabilities, "Always move nodes."
  )
      ->capture_default_str();
  lp->add_flag(
        "--r-lp-scale-batches-with-threads",
        ctx.refinement.lp.scale_chunks_with_threads,
        "Scale the number of synchronization rounds with the number of threads."
  )
      ->capture_default_str();

  return lp;
}

CLI::Option_group *create_colored_lp_refinement_options(CLI::App *app, Context &ctx) {
  auto *lp = app->add_option_group("Refinement -> Colored Label Propagation");

  lp->add_option(
        "--r-clp-num-iterations",
        ctx.refinement.colored_lp.num_iterations,
        "Number of label propagation iterations."
  )
      ->capture_default_str();
  lp->add_option("--r-clp-commit-strategy", ctx.refinement.colored_lp.move_execution_strategy)
      ->transform(
          CLI::CheckedTransformer(get_label_propagation_move_execution_strategies()).description("")
      )
      ->description(R"(Strategy to decide which moves to execute:
  - probabilistic: Assign each node a probability based on its gain and execute random moves accordingly 
  - best:          Identify the best global moves and execute them
  - local:         Execute all local moves, risking an imbalanced partition)")
      ->capture_default_str();
  lp->add_option(
        "--r-clp-num-commit-rounds",
        ctx.refinement.colored_lp.num_move_execution_iterations,
        "Number of move commitment rounds."
  )
      ->capture_default_str();
  lp->add_option(
        "--r-clp-num-attempts",
        ctx.refinement.colored_lp.num_probabilistic_move_attempts,
        "[commit-strategy=probabilistic] Number of attempts to use the "
        "probabilistic commitment strategy before giving up."
  )
      ->capture_default_str();
  lp->add_flag(
        "--r-clp-sort-by-rel-gain",
        ctx.refinement.colored_lp.sort_by_rel_gain,
        "[commit-strategy=best] Sort move candidates by their relative gain "
        "rather than their absolute gain."
  )
      ->capture_default_str();
  lp->add_flag("--r-clp-track-block-weights", ctx.refinement.colored_lp.track_local_block_weights)
      ->capture_default_str();

  // Control number of coloring supersteps
  lp->add_option("--r-clp-max-num-chunks", ctx.refinement.colored_lp.max_num_coloring_chunks)
      ->capture_default_str();
  lp->add_option("--r-clp-min-num-chunks", ctx.refinement.colored_lp.min_num_coloring_chunks)
      ->capture_default_str();
  lp->add_option(
        "--r-clp-num-chunks",
        ctx.refinement.colored_lp.fixed_num_coloring_chunks,
        "Number of supersteps of the coloring algorithm. If set to 0, the "
        "value is derived from the min and max bounds."
  )
      ->capture_default_str();
  lp->add_flag(
        "--r-clp-scale-chunks-with-threads",
        ctx.refinement.colored_lp.scale_coloring_chunks_with_threads
  )
      ->capture_default_str();
  lp->add_option(
        "--r-clp-small-color-blacklist",
        ctx.refinement.colored_lp.small_color_blacklist,
        "Sort colors by their size, then exclude the smallest colors such that "
        "roughly <param>% of all nodes are excluded"
  )
      ->capture_default_str();
  lp->add_flag(
        "--r-clp-only-blacklist-on-input-level",
        ctx.refinement.colored_lp.only_blacklist_input_level,
        "Only blacklist nodes when refining the input graph."
  )
      ->capture_default_str();

  lp->add_flag(
      "--r-clp-active-set", ctx.refinement.colored_lp.use_active_set, "Enable active set strategy."
  );

  return lp;
}

CLI::Option_group *create_greedy_balancer_options(CLI::App *app, Context &ctx) {
  auto *balancer = app->add_option_group("Refinement -> Balancer");

  balancer->add_option("--r-b-max-num-rounds", ctx.refinement.greedy_balancer.max_num_rounds)
      ->capture_default_str();
  balancer
      ->add_flag(
          "--r-b-enable-strong-balancing", ctx.refinement.greedy_balancer.enable_strong_balancing
      )
      ->capture_default_str();
  balancer
      ->add_option(
          "--r-b-nodes-per-block",
          ctx.refinement.greedy_balancer.num_nodes_per_block,
          "Number of nodes selected for each overloaded block on each PE."
      )
      ->capture_default_str();
  balancer
      ->add_flag(
          "--r-b-enable-fast-balancing", ctx.refinement.greedy_balancer.enable_fast_balancing
      )
      ->capture_default_str();
  balancer
      ->add_option(
          "--r-b-fast-balancing-threshold",
          ctx.refinement.greedy_balancer.fast_balancing_threshold,
          "Perform a fast balancing round if strong balancing improved the imbalance by less than "
          "this value, e.g., 0.01 for 1%."
      )
      ->capture_default_str();

  return balancer;
}

CLI::Option_group *create_coarsening_options(CLI::App *app, Context &ctx) {
  auto *coarsening = app->add_option_group("Coarsening");

  coarsening
      ->add_option(
          "-C,--c-contraction-limit", ctx.coarsening.contraction_limit, "Contraction limit."
      )
      ->capture_default_str();
  coarsening
      ->add_option(
          "--c-max-cluster-weight-multiplier",
          ctx.coarsening.cluster_weight_multiplier,
          "Multiplier for the maximum cluster weight."
      )
      ->capture_default_str();
  coarsening
      ->add_option(
          "--c-max-local-levels",
          ctx.coarsening.max_local_clustering_levels,
          "Maximum number of local clustering levels."
      )
      ->capture_default_str();
  coarsening
      ->add_option(
          "--c-max-global-levels",
          ctx.coarsening.max_global_clustering_levels,
          "Maximum number of global clustering levels."
      )
      ->capture_default_str();
  coarsening
      ->add_option("--c-local-clustering-algorithm", ctx.coarsening.local_clustering_algorithm)
      ->transform(CLI::CheckedTransformer(get_local_clustering_algorithms()).description(""))
      ->description(R"(Local clustering algorithm, options are:
  - noop: disable local clustering
  - lp:   parallel label propagation)")
      ->capture_default_str();
  coarsening
      ->add_option("--c-global-clustering-algorithm", ctx.coarsening.global_clustering_algorithm)
      ->transform(CLI::CheckedTransformer(get_global_clustering_algorithms()).description(""))
      ->description(R"(Global clustering algorithm, options are:
  - noop:           disable global clustering
  - lp:             parallel label propagation without active set strategy
  - active-set-lp:  parallel label propagation with active set strategy
  - locking-lp:     parallel label propagation with cluster-join requests
  - hem:            heavy edge matching
  - hem-lp:         heavy edge matching + label propagation)")

      ->capture_default_str();
  coarsening->add_option(
      "--c-max-cnode-imbalance",
      ctx.coarsening.max_cnode_imbalance,
      "Maximum coarse node imbalance before rebalancing cluster assignment."
  );
  coarsening->add_flag(
      "--c-migrate-cnode-prefix",
      ctx.coarsening.migrate_cnode_prefix,
      "Migrate the first few nodes of overloaded PEs rather than the last few."
  );
  coarsening->add_flag(
      "--c-force-perfect-cnode-balance",
      ctx.coarsening.force_perfect_cnode_balance,
      "If imbalance threshold is exceeded, migrate nodes until perfectly "
      "balanced."
  );

  create_global_lp_coarsening_options(app, ctx);
  create_local_lp_coarsening_options(app, ctx);
  create_hem_coarsening_options(app, ctx);

  return coarsening;
}

CLI::Option_group *create_global_lp_coarsening_options(CLI::App *app, Context &ctx) {
  auto *lp = app->add_option_group("Coarsening -> Global Label Propagation");

  lp->add_option(
        "--c-glp-iterations", ctx.coarsening.global_lp.num_iterations, "Number of iterations."
  )
      ->capture_default_str();
  lp->add_option(
        "--c-glp-total-chunks",
        ctx.coarsening.global_lp.total_num_chunks,
        "Number of synchronization rounds times number of PEs."
  )
      ->capture_default_str();
  lp->add_option(
        "--c-glp-min-chunks",
        ctx.coarsening.global_lp.min_num_chunks,
        "Minimum number of synchronization rounds."
  )
      ->capture_default_str();
  lp->add_option(
        "--c-glp-num-chunks",
        ctx.coarsening.global_lp.fixed_num_chunks,
        "Set the number of chunks to a fixed number rather than deducing it "
        "from other parameters (0 = deduce)."
  )
      ->capture_default_str();
  lp->add_option(
        "--c-glp-active-large-degree-threshold",
        ctx.coarsening.global_lp.active_high_degree_threshold,
        "Do not move nodes with degree larger than this."
  )
      ->capture_default_str();
  lp->add_option(
        "--c-glp-passive-large-degree-threshold",
        ctx.coarsening.global_lp.passive_high_degree_threshold,
        "Do not look at nodes with a degree larger than this when moving other "
        "nodes."
  )
      ->capture_default_str();
  lp->add_flag(
        "--c-glp-scale-batches-with-threads",
        ctx.coarsening.global_lp.scale_chunks_with_threads,
        "Scale the number of synchronization rounds with the number of threads."
  )
      ->capture_default_str();
  lp->add_flag("--c-glp-sync-cluster-weights", ctx.coarsening.global_lp.sync_cluster_weights);
  lp->add_flag("--c-glp-enforce-cluster-weights", ctx.coarsening.global_lp.enforce_cluster_weights);
  lp->add_flag("--c-glp-cheap-toplevel", ctx.coarsening.global_lp.cheap_toplevel);
  lp->add_flag("--c-glp-prevent-cyclic-moves", ctx.coarsening.global_lp.prevent_cyclic_moves);
  lp->add_flag("--c-glp-enforce-legacy-weight", ctx.coarsening.global_lp.enforce_legacy_weight);

  return lp;
}

CLI::Option_group *create_local_lp_coarsening_options(CLI::App *app, Context &ctx) {
  auto *lp = app->add_option_group("Coarsening -> Local Label Propagation");

  lp->add_option(
        "--c-llp-iterations", ctx.coarsening.local_lp.num_iterations, "Number of iterations."
  )
      ->capture_default_str();
  lp->add_flag(
        "--c-llp-ignore-ghost-nodes",
        ctx.coarsening.local_lp.ignore_ghost_nodes,
        "Ignore ghost nodes for cluster rating."
  )
      ->capture_default_str();
  lp->add_flag(
        "--c-llp-keep-ghost-clusters",
        ctx.coarsening.local_lp.keep_ghost_clusters,
        "Keep clusters of ghost clusters and remap them to local cluster IDs."
  )
      ->capture_default_str();

  return lp;
}

CLI::Option_group *create_hem_coarsening_options(CLI::App *app, Context &ctx) {
  auto *lp = app->add_option_group("Coarsening -> Heavy Edge Matching");

  // Control number of coloring supersteps
  lp->add_option("--c-hem-max-num-chunks", ctx.coarsening.hem.max_num_coloring_chunks)
      ->capture_default_str();
  lp->add_option("--c-hem-min-num-chunks", ctx.coarsening.hem.min_num_coloring_chunks)
      ->capture_default_str();
  lp->add_option(
        "--c-hem-num-chunks",
        ctx.coarsening.hem.fixed_num_coloring_chunks,
        "Number of supersteps of the coloring algorithm. If set to 0, the "
        "value is derived from the min and max bounds."
  )
      ->capture_default_str();
  lp->add_flag(
        "--c-hem-scale-chunks-with-threads", ctx.coarsening.hem.scale_coloring_chunks_with_threads
  )
      ->capture_default_str();
  lp->add_option(
        "--c-hem-small-color-blacklist",
        ctx.coarsening.hem.small_color_blacklist,
        "Sort colors by their size, then exclude the smallest colors such that "
        "roughly <param>% of all nodes are excluded"
  )
      ->capture_default_str();
  lp->add_flag(
        "--c-hem-only-blacklist-on-input-level",
        ctx.coarsening.hem.only_blacklist_input_level,
        "Only blacklist nodes when refining the input graph."
  )
      ->capture_default_str();
  lp->add_flag(
      "--c-hem-ignore-weight-limit",
      ctx.coarsening.hem.ignore_weight_limit,
      "Ignore cluster weight limit."
  );

  return lp;
}

CLI::Option_group *create_jet_refinement_options(CLI::App *app, Context &ctx) {
  auto *jet = app->add_option_group("Refinement -> JET");

  jet->add_option("--r-jet-num-iterations", ctx.refinement.jet.num_iterations)
      ->capture_default_str();
  jet->add_option("--r-jet-min-c", ctx.refinement.jet.min_c)->capture_default_str();
  jet->add_option("--r-jet-max-c", ctx.refinement.jet.max_c)->capture_default_str();
  jet->add_flag("--r-jet-interpolate-c", ctx.refinement.jet.interpolate_c)->capture_default_str();
  jet->add_flag("--r-jet-use-abortion-threshold", ctx.refinement.jet.use_abortion_threshold)
      ->capture_default_str();
  jet->add_option("--r-jet-abortion-threshold", ctx.refinement.jet.abortion_threshold)
      ->capture_default_str();

  return jet;
}
} // namespace kaminpar::dist
