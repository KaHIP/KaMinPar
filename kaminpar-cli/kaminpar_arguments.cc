/*******************************************************************************
 * Command line arguments for the shared-memory partitioner.
 *
 * @file:   kaminpar_arguments.cc
 * @author: Daniel Seemaier
 * @date:   14.10.2022
 ******************************************************************************/
#include "kaminpar-cli/kaminpar_arguments.h"

#include "kaminpar-cli/CLI11.h"

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/context_io.h"

namespace kaminpar::shm {
void create_all_options(CLI::App *app, Context &ctx) {
  create_graph_compression_options(app, ctx);
  create_partitioning_options(app, ctx);
  create_debug_options(app, ctx);
  create_coarsening_options(app, ctx);
  create_initial_partitioning_options(app, ctx);
  create_refinement_options(app, ctx);
}

CLI::Option_group *create_graph_compression_options(CLI::App *app, Context &ctx) {
  auto *compression = app->add_option_group("Graph Compression");

  compression->add_flag("-c,--compress", ctx.compression.enabled, "Enable graph compression")
      ->default_val(false);
  compression
      ->add_flag(
          "--may-dismiss",
          ctx.compression.may_dismiss,
          "Whether the compressed graph is only used if it uses less memory than the uncompressed "
          "graph."
      )
      ->default_val(false);

  return compression;
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

  // Partitioning options
  partitioning->add_option("-m,--p-mode", ctx.partitioning.mode)
      ->transform(CLI::CheckedTransformer(get_partitioning_modes()).description(""))
      ->description(R"(Partitioning scheme:
  - deep: deep multilevel
  - rb:   recursive multilevel bipartitioning
  - kway: k-way multilevel with rb for initial partitioning)")
      ->capture_default_str();
  partitioning
      ->add_option(
          "--p-deep-initial-partitioning-mode", ctx.partitioning.deep_initial_partitioning_mode
      )
      ->transform(CLI::CheckedTransformer(get_initial_partitioning_modes()).description(""))
      ->description(R"(Chooses the initial partitioning mode:
  - sequential:     do not diversify initial partitioning by replicating coarse graphs
  - async-parallel: diversify initial partitioning by replicating coarse graphs each branch of the replication tree asynchronously
  - sync-parallel:  same as async-parallel, but process branches synchronously)")
      ->capture_default_str();
  partitioning->add_option(
      "--p-deep-initial-partitioning-load",
      ctx.partitioning.deep_initial_partitioning_load,
      "Fraction of cores that should be used for the coarse graph replication phase of deep MGP. A "
      "value of '1' will replicate the graph once for every PE, whereas smaller values lead to "
      "fewer replications."
  );

  create_partitioning_rearrangement_options(app, ctx);

  return partitioning;
}

CLI::Option_group *create_partitioning_rearrangement_options(CLI::App *app, Context &ctx) {
  auto *rearrangement = app->add_option_group("Partitioning -> Rearrangement");

  rearrangement->add_option("--node-order", ctx.node_ordering)
      ->transform(CLI::CheckedTransformer(get_node_orderings()).description(""))
      ->description(R"(Criteria by which the nodes of the graph are sorted and rearranged:
  - natural:     keep node order of the graph (do not rearrange)
  - deg-buckets: sort nodes by degree bucket and rearrange accordingly
  - implicit-deg-buckets: nodes of the input graph are sorted by deg-buckets order)")
      ->capture_default_str();
  rearrangement->add_option("--edge-order", ctx.edge_ordering)
      ->transform(CLI::CheckedTransformer(get_edge_orderings()).description(""))
      ->description(R"(Criteria by which the edges of the graph are sorted and rearranged:
  - natural:     keep edge order of the graph (do not rearrange)
  - compression: sort the edges of each neighbourhood with the ordering of the corresponding compressed graph)"
      )
      ->capture_default_str();

  return rearrangement;
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
  create_contraction_coarsening_options(app, ctx);

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

  lp->add_option(
        "--c-lp-two-phases",
        ctx.coarsening.lp.use_two_phases,
        "Uses two phases in each iteration, where in the second phase the high-degree nodes are "
        "treated separately"
  )
      ->default_val(true)
      ->capture_default_str();
  lp->add_option("--c-lp-second-phase-select-mode", ctx.coarsening.lp.second_phase_select_mode)
      ->transform(CLI::CheckedTransformer(get_second_phase_select_modes()).description(""))
      ->description(
          R"(Determines the mode for selecting nodes for the second phase of label propagation.
Options are:
  - high-degree:     Select nodes with high degree
  - full-rating-map: Select nodes which have a full rating map in the first phase
  )"
      )
      ->capture_default_str();
  lp->add_option(
        "--c-lp-second-phase-aggregation-mode", ctx.coarsening.lp.second_phase_aggregation_mode
  )
      ->transform(CLI::CheckedTransformer(get_second_phase_aggregation_modes()).description(""))
      ->description(
          R"(Determines the mode for aggregating ratings in the second phase of label propagation.
Options are:
  - direct:   Write the ratings directly into the global vector (shared between threads)
  - buffered: Write the ratings into a thread-local buffer and then copy them into the global vector when the buffer is full
  )"
      )
      ->capture_default_str();

  lp->add_option("--c-lp-isolated-nodes-strategy", ctx.coarsening.lp.isolated_nodes_strategy)
      ->transform(
          CLI::CheckedTransformer(get_isolated_nodes_clustering_strategies()).description("")
      )
      ->description(R"(Determines the strategy for handling isolated nodes during graph clustering.
Options are:
  - keep:                   Keep isolated nodes in the graph
  - match-always:           Pack pairs of isolated nodes into the same cluster (respecting the maximum cluster weight limit)
  - cluster-always:         Pack any number of isolated nodes into the same cluster (respecting the maximum cluster weight limit)
  - match-during-two-hop:   Only match isolated nodes after two-hop clustering was triggered
  - cluster-during-two-hop: Only cluster isolated nodes after two-hop clustering was triggered
  )")
      ->capture_default_str();

  return lp;
}

CLI::Option_group *create_contraction_coarsening_options(CLI::App *app, Context &ctx) {
  auto *contraction = app->add_option_group("Coarsening -> Contraction");

  contraction
      ->add_option(
          "--c-con-edge-buffer",
          ctx.coarsening.contraction.use_edge_buffer,
          "Whether to use a temporary buffer for storing the contracted edges or to compute the "
          "contracted edges twice"
      )
      ->capture_default_str();

  return contraction;
}

CLI::Option_group *create_initial_partitioning_options(CLI::App *app, Context &ctx) {
  auto *ip = app->add_option_group("Initial Partitioning");

  ip->add_flag(
        "--i-r-disable", ctx.initial_partitioning.refinement.disabled, "Disable initial refinement."
  )
      ->capture_default_str();

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
  fm->add_option(
        "--r-fm-abortion-threshold",
        ctx.refinement.kway_fm.abortion_threshold,
        "Stop FM iterations if the edge cut reduction of the previous "
        "iteration falls below this threshold (lower = weaker, but faster)."
  )
      ->capture_default_str();
  fm->add_option("--r-fm-gain-cache", ctx.refinement.kway_fm.gain_cache_strategy)
      ->transform(CLI::CheckedTransformer(get_gain_cache_strategies()).description(""))
      ->capture_default_str();
  fm->add_option(
        "--r-fm-constant-high-degree-threshold",
        ctx.refinement.kway_fm.constant_high_degree_threshold
  )
      ->capture_default_str();
  fm->add_option(
        "--r-fm-k-based-high-degree-threshold", ctx.refinement.kway_fm.k_based_high_degree_threshold
  )
      ->capture_default_str();
  fm->add_flag("--r-fm-preallocate-gain-cache", ctx.refinement.kway_fm.preallocate_gain_cache)
      ->capture_default_str();
  fm->add_flag(
        "--r-fm-dbg-batch-size-stats", ctx.refinement.kway_fm.dbg_compute_batch_size_statistics
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

  debug->add_option("--d-dump-graph-filename", ctx.debug.dump_graph_filename)
      ->capture_default_str();
  debug->add_option("--d-dump-partition-filename", ctx.debug.dump_partition_filename)
      ->capture_default_str();

  debug
      ->add_flag(
          "--d-dump-toplevel-graph",
          ctx.debug.dump_toplevel_graph,
          "Write the toplevel graph to disk. Note that this graph might be different from the "
          "input graph, as isolated nodes might have been removed and nodes might have been "
          "reordered."
      )
      ->capture_default_str();

  debug
      ->add_flag(
          "--d-dump-toplevel-partition",
          ctx.debug.dump_toplevel_partition,
          "Write the partition of the toplevel graph before- and after running refinement to disk. "
          "This partition should only be used together with the toplevel graph obtained via "
          "--d-dump-toplevel-graph."
      )
      ->capture_default_str();
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
        ctx.debug.dump_toplevel_graph = true;
        ctx.debug.dump_toplevel_partition = true;
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
