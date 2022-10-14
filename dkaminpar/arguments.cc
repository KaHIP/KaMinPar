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

namespace kaminpar::dist {
void create_all_options(CLI::App* app, Context& ctx) {
    create_partitioning_options(app, ctx);
    create_debug_options(app, ctx);
    create_initial_partitioning_options(app, ctx);
    create_refinement_options(app, ctx);
    create_fm_refinement_options(app, ctx);
    create_lp_refinement_options(app, ctx);
}

CLI::Option_group* create_partitioning_options(CLI::App* app, Context& ctx) {
    auto* partitioning = app->add_option_group("Partitioning");

    partitioning->add_option("-e,--epsilon", ctx.partition.epsilon, "Maximum allowed block imbalance.")
        ->check(CLI::NonNegativeNumber)
        ->configurable(false)
        ->capture_default_str();
    partitioning
        ->add_option(
            "-K,--block-multiplier", ctx.partition.k_prime,
            "Maximum block count with which the initial partitioner is called."
        )
        ->capture_default_str();
    partitioning->add_option("-m,--mode", ctx.partition.mode)
        ->transform(CLI::CheckedTransformer(get_partitioning_modes()).description(""))
        ->description(R"(Partitioning scheme, possible options are:
  - deep: distributed deep multilevel graph partitioning
  - kway: direct k-way multilevel graph partitioning)")
        ->capture_default_str();

    return partitioning;
}

CLI::Option_group *create_debug_options(CLI::App *app, Context &ctx) {
    auto *debug = app->add_option_group("Debug");

    debug->add_flag("--d-save-imbalanced-partitions", ctx.debug.save_imbalanced_partitions)
        ->configurable(false)
        ->capture_default_str();
    debug->add_flag("--d-save-graph-hierarchy", ctx.debug.save_graph_hierarchy)
        ->configurable(false)
        ->capture_default_str();
    debug->add_flag("--d-save-coarsest-graph", ctx.debug.save_coarsest_graph)
        ->configurable(false)
        ->capture_default_str();
    debug->add_flag("--d-save-clustering-hierarchy", ctx.debug.save_clustering_hierarchy)
        ->configurable(false)
        ->capture_default_str();

    return debug;
}

CLI::Option_group *create_initial_partitioning_options(CLI::App *app, Context &ctx) {
    auto *ip = app->add_option_group("Initial Partitioning");

    ip->add_option("--i-algorithm", ctx.initial_partitioning.algorithm)
        ->check(CLI::CheckedTransformer(get_initial_partitioning_algorithms()).description(""))
        ->description(R"(Algorithm used for initial partitioning. Options are:
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
} // namespace kaminpar::dist
