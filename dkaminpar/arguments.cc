/*******************************************************************************
 * @file:   arguments.cc
 * @author: Daniel Seemaier
 * @date:   15.10.2022
 * @brief:  Command line arguments for the distributed partitioner.
 ******************************************************************************/
// clang-format off
#include "common/CLI11.h"
// clang-on 

#include "dkaminpar/arguments.h"

#include "dkaminpar/context.h"
#include "dkaminpar/context_io.h"

namespace kaminpar::dist {
void create_all_options(CLI::App* app, Context& ctx) {
    create_partitioning_options(app, ctx);
    create_debug_options(app, ctx);
    create_coarsening_options(app, ctx);
    create_global_lp_coarsening_options(app, ctx);
    create_local_lp_coarsening_options(app, ctx);
    create_initial_partitioning_options(app, ctx);
    create_refinement_options(app, ctx);
    create_fm_refinement_options(app, ctx);
    create_lp_refinement_options(app, ctx);
    create_balancer_options(app, ctx);
}

CLI::Option_group* create_partitioning_options(CLI::App* app, Context& ctx) {
    auto* partitioning = app->add_option_group("Partitioning");

    partitioning->add_option("-e,--epsilon", ctx.partition.epsilon, "Maximum allowed block imbalance.")
        ->check(CLI::NonNegativeNumber)
        ->configurable(false)
        ->capture_default_str();
    partitioning
        ->add_option(
            "-K,--block-multiplier", ctx.partition.K,
            "Maximum block count with which the initial partitioner is called."
        )
        ->capture_default_str();
    partitioning->add_option("-m,--mode", ctx.partition.mode)
        ->transform(CLI::CheckedTransformer(get_partitioning_modes()).description(""))
        ->description(R"(Partitioning scheme, possible options are:
  - deep: distributed deep multilevel graph partitioning
  - deeper: distributed deep multilevel graph partitioning with optional PE splitting and graph replication
  - kway: direct k-way multilevel graph partitioning)")
        ->capture_default_str();
    partitioning->add_flag("--enable-pe-splitting", ctx.partition.enable_pe_splitting, "Enable PE splitting and graph replication in deep MGP")
        ->capture_default_str();

    return partitioning;
}

CLI::Option_group *create_debug_options(CLI::App *app, Context &ctx) {
    auto *debug = app->add_option_group("Debug");

    debug->add_flag("--d-save-finest-graph", ctx.debug.save_finest_graph)
        ->configurable(false)
        ->capture_default_str();
    debug->add_flag("--d-save-coarsest-graph", ctx.debug.save_coarsest_graph)
        ->configurable(false)
        ->capture_default_str();
    debug->add_flag("--d-save-graph-hierarchy", ctx.debug.save_graph_hierarchy)
        ->configurable(false)
        ->capture_default_str();
    debug->add_flag("--d-save-clustering-hierarchy", ctx.debug.save_clustering_hierarchy)
        ->configurable(false)
        ->capture_default_str();
    debug->add_flag("--d-save-partition-hierarchy", ctx.debug.save_partition_hierarchy)
        ->configurable(false)
        ->capture_default_str();

    return debug;
}

CLI::Option_group *create_initial_partitioning_options(CLI::App *app, Context &ctx) {
    auto *ip = app->add_option_group("Initial Partitioning");

    ip->add_option("--i-algorithm", ctx.initial_partitioning.algorithm)
        ->check(CLI::CheckedTransformer(get_initial_partitioning_algorithms()).description(""))
        ->description(R"(Algorithm used for initial partitioning. Options are:
  - random:    assign nodes to blocks randomly
  - kaminpar:  use KaMinPar for initial partitioning
  - mtkahypar: use Mt-KaHyPar for inital partitioning)")
        ->capture_default_str();
    ip->add_option("--i-mtkahypar-preset", ctx.initial_partitioning.mtkahypar.preset_filename, "Preset configuration file for Mt-KaHyPar when using it for initial partitioning.")
        ->check(CLI::ExistingFile)
        ->capture_default_str();

    return ip;
}

CLI::Option_group *create_refinement_options(CLI::App *app, Context &ctx) {
    auto *refinement = app->add_option_group("Refinement");

    refinement->add_option("--r-algorithm", ctx.refinement.algorithm)
        ->transform(CLI::CheckedTransformer(get_kway_refinement_algorithms()).description(""))
        ->description(R"(K-way refinement algorithm. Possible options are:
  - noop:        disable k-way refinement
  - lp:          distributed label propagation
  - local-fm:    PE-local FM
  - fm:          distributed FM
  - lp+local-fm: distributed label propagation -> PE-local fm
  - lp+fm:       distributed label propagation -> distributed fm)")
        ->capture_default_str();
    refinement->add_flag("--r-refine-coarsest-graph", ctx.refinement.refine_coarsest_level, "Also run the refinement algorithms on the coarsest graph.")
        ->capture_default_str();

    return refinement;
}

CLI::Option_group *create_fm_refinement_options(CLI::App *app, Context &ctx) {
    auto *fm = app->add_option_group("Refinement -> FM");

    fm->add_option("--r-fm-alpha", ctx.refinement.fm.alpha, "Alpha parameter for the adaptive stopping rule.")
        ->capture_default_str();
    fm->add_option("--r-fm-radius", ctx.refinement.fm.radius, "Radius for the search graphs.")
        ->capture_default_str();
    fm->add_option("--r-fm-hops", ctx.refinement.fm.pe_radius, "Number of PE hops for the BFS search.")
        ->capture_default_str();
    fm->add_flag("--r-fm-overlap-regions", ctx.refinement.fm.overlap_regions, "Allow search regions to overlap.")
        ->capture_default_str();
    fm->add_option("--r-fm-iterations", ctx.refinement.fm.num_iterations, "Number of FM iterations.")
        ->capture_default_str();
    fm->add_flag("--r-fm-sequential", ctx.refinement.fm.sequential, "Refine search graphs sequentially.")
        ->capture_default_str();
    fm->add_flag("--r-fm-premove-locally", ctx.refinement.fm.premove_locally, "Move nodes right away, i.e., before global synchronization steps.")
        ->capture_default_str();
    fm->add_option("--r-fm-bound-degree", ctx.refinement.fm.bound_degree, "Add at most this many neighbors of a high-degree node to the search region.")
        ->capture_default_str();
    fm->add_flag("--r-fm-contract-border", ctx.refinement.fm.contract_border, "Contract the exterior of the search graph")
        ->capture_default_str();

    return fm;
}

CLI::Option_group *create_lp_refinement_options(CLI::App *app, Context &ctx) {
    auto *lp = app->add_option_group("Refinement -> LP");

    lp->add_option("--r-lp-iterations", ctx.refinement.lp.num_iterations, "Number of label propagation iterations.")
        ->capture_default_str();
    lp->add_option("--r-lp-total-chunks", ctx.refinement.lp.total_num_chunks, "Number of synchronization rounds times number of PEs.")
        ->capture_default_str();
    lp->add_option("--r-lp-min-chunks", ctx.refinement.lp.min_num_chunks, "Minimum number of synchronization rounds.")
        ->capture_default_str();
    lp->add_option("--r-lp-active-large-degree-threshold", ctx.refinement.lp.active_high_degree_threshold, "Do not move nodes with degree larger than this.")
        ->capture_default_str();
    lp->add_flag("--r-lp-ignore-probabilities", ctx.refinement.lp.ignore_probabilities, "Always move nodes.")
        ->capture_default_str();
    lp->add_flag("--r-lp-scale-batches-with-threads", ctx.refinement.lp.scale_chunks_with_threads, "Scale the number of synchronization rounds with the number of threads.")
        ->capture_default_str();

    return lp;
}

CLI::Option_group *create_balancer_options(CLI::App *app, Context &ctx) {
    auto *balancer = app->add_option_group("Refinement -> Balancer");

    balancer->add_option("--r-b-algorithm", ctx.refinement.balancing.algorithm)
        ->transform(CLI::CheckedTransformer(get_balancing_algorithms()).description(""))
        ->description(R"(Balancing algorithm, options are:
  - distributed: distributed balancing algorithm)")
        ->capture_default_str();
    balancer->add_option("--r-b-nodes-per-block", ctx.refinement.balancing.num_nodes_per_block, "Number of nodes selected for each overloaded block on each PE.")
        ->capture_default_str();

    return balancer;
}

CLI::Option_group *create_coarsening_options(CLI::App *app, Context &ctx) {
    auto *coarsening = app->add_option_group("Coarsening");

    coarsening->add_option("-C,--c-contraction-limit", ctx.coarsening.contraction_limit, "Contraction limit.")
        ->capture_default_str();
    coarsening->add_option("--c-max-cluster-weight-multiplier", ctx.coarsening.cluster_weight_multiplier, "Multiplier for the maximum cluster weight.")
        ->capture_default_str();
    coarsening->add_option("--c-max-local-levels", ctx.coarsening.max_local_clustering_levels, "Maximum number of local clustering levels.")
        ->capture_default_str();
    coarsening->add_option("--c-max-global-levels", ctx.coarsening.max_global_clustering_levels, "Maximum number of global clustering levels.")
        ->capture_default_str();
    coarsening->add_option("--c-local-clustering-algorithm", ctx.coarsening.local_clustering_algorithm)
        ->transform(CLI::CheckedTransformer(get_local_clustering_algorithms()).description(""))
        ->description(R"(Local clustering algorithm, options are:
  - noop: disable local clustering
  - lp:   parallel label propagation)")
        ->capture_default_str();
    coarsening->add_option("--c-global-clustering-algorithm", ctx.coarsening.global_clustering_algorithm)
        ->transform(CLI::CheckedTransformer(get_global_clustering_algorithms()).description(""))
        ->description(R"(Global clustering algorithm, options are:
  - noop:           disable global clustering
  - lp:             parallel label propagation without active set strategy
  - active--set-lp: parallel label propagation with active set strategy
  - locking-lp:     parallel label propagation with cluster-join requests)")
        ->capture_default_str();
    coarsening->add_option("--c-global-contraction-algorithm", ctx.coarsening.global_contraction_algorithm)
        ->transform(CLI::CheckedTransformer(get_global_contraction_algorithms()).description(""))
        ->description(R"(Algorithm to contract a global clustering, options are:
  - no-migration:      do not redistribute any nodes
  - minimal-migration: only redistribute coarse nodes s.t. each PE has the same number of nodes
  - full-migration:    redistribute all coarse nodes round-robin)")
        ->capture_default_str();
    
    return coarsening;
}

CLI::Option_group *create_global_lp_coarsening_options(CLI::App *app, Context &ctx) {
    auto *lp = app->add_option_group("Coarsening -> Global Label Propagation");

    lp->add_option("--c-glp-iterations", ctx.coarsening.global_lp.num_iterations, "Number of iterations.")
        ->capture_default_str();
    lp->add_option("--c-glp-total-chunks", ctx.coarsening.global_lp.total_num_chunks, "Number of synchronization rounds times number of PEs.")
        ->capture_default_str();
    lp->add_option("--c-glp-min-chunks", ctx.coarsening.global_lp.min_num_chunks, "Minimum number of synchronization rounds.")
        ->capture_default_str();
    lp->add_option("--c-glp-num-chunks", ctx.coarsening.global_lp.num_chunks, "Set the number of chunks to a fixed number rather than deducing it from other parameters (0 = deduce).")
        ->capture_default_str();
    lp->add_option("--c-glp-active-large-degree-threshold", ctx.coarsening.global_lp.active_high_degree_threshold, "Do not move nodes with degree larger than this.")
        ->capture_default_str();
    lp->add_option("--c-glp-passive-large-degree-threshold", ctx.coarsening.global_lp.passive_high_degree_threshold, "Do not look at nodes with a degree larger than this when moving other nodes.")
        ->capture_default_str();
    lp->add_flag("--c-glp-scale-batches-with-threads", ctx.coarsening.global_lp.scale_chunks_with_threads, "Scale the number of synchronization rounds with the number of threads.")
        ->capture_default_str();

    return lp;
}

CLI::Option_group *create_local_lp_coarsening_options(CLI::App *app, Context &ctx) {
    auto *lp = app->add_option_group("Coarsening -> Local Label Propagation");

    lp->add_option("--c-llp-iterations", ctx.coarsening.local_lp.num_iterations, "Number of iterations.")
        ->capture_default_str();
    lp->add_flag("--c-llp-ignore-ghost-nodes", ctx.coarsening.local_lp.ignore_ghost_nodes, "Ignore ghost nodes for cluster rating.")
        ->capture_default_str();
    lp->add_flag("--c-llp-keep-ghost-clusters", ctx.coarsening.local_lp.keep_ghost_clusters, "Keep clusters of ghost clusters and remap them to local cluster IDs.")
        ->capture_default_str();

    return lp;
}
} // namespace kaminpar::dist
