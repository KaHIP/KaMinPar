/*******************************************************************************
 * @file:   presets.cc
 * @author: Daniel Seemaier
 * @date:   15.10.2022
 * @brief:  Configuration presets.
 ******************************************************************************/
#include "dkaminpar/presets.h"

#include <stdexcept>

#include "dkaminpar/context.h"

#include "kaminpar/presets.h"

namespace kaminpar::dist {
Context create_context_by_preset_name(const std::string& name) {
    if (name == "default" || name == "fast") {
        return create_default_context();
    } else if (name == "strong") {
        return create_strong_context();
    } else if (name == "ipdps23-submission-default" || name == "ipdps23-submission-fast") {
        return create_ipdps23_submission_default_context();
    } else if (name == "ipdps23-submission-strong") {
        return create_ipdps23_submission_strong_context();
    }

    throw std::runtime_error("invalid preset name");
}

std::unordered_set<std::string> get_preset_names() {
    return {
        "default",
        "fast",
        "strong",
        "ipdps23-submission-default",
        "ipdps23-submission-fast",
        "ipdps23-submission-strong",
    };
}

Context create_default_context() {
    return {
        .seed         = 0,
        .rearrange_by = GraphOrdering::DEGREE_BUCKETS,
        .partition =
            {
                .k                     = 0,
                .K                     = 128,
                .epsilon               = 0.03,
                .mode                  = PartitioningMode::DEEP,
                .enable_pe_splitting   = true,
                .simulate_singlethread = true,
                .graph                 = GraphContext(),
            },
        .parallel =
            {
                .num_threads                     = 1,
                .num_mpis                        = 1,
                .use_interleaved_numa_allocation = true,
            },
        .coarsening =
            {
                .max_global_clustering_levels = std::numeric_limits<std::size_t>::max(),
                .global_clustering_algorithm  = GlobalClusteringAlgorithm::LP,
                .global_contraction_algorithm = GlobalContractionAlgorithm::MINIMAL_MIGRATION,
                .global_lp =
                    {
                        .num_iterations                       = 3,
                        .passive_high_degree_threshold        = 1'000'000,
                        .active_high_degree_threshold         = 1'000'000,
                        .max_num_neighbors                    = kInvalidNodeID,
                        .merge_singleton_clusters             = true,
                        .merge_nonadjacent_clusters_threshold = 0.5,
                        .total_num_chunks                     = 128,
                        .num_chunks                           = 0,
                        .min_num_chunks                       = 8,
                        .ignore_ghost_nodes                   = false, // unused
                        .keep_ghost_clusters                  = false,
                        .scale_chunks_with_threads            = false,
                    },
                .hem =
                    {
                        .num_coloring_chunks                = 0,
                        .max_num_coloring_chunks            = 128,
                        .min_num_coloring_chunks            = 8,
                        .scale_coloring_chunks_with_threads = false,
                        .small_color_blacklist              = 0,
                        .only_blacklist_input_level         = false,
                        .ignore_weight_limit                = false,
                    },
                .max_local_clustering_levels = 0,
                .local_clustering_algorithm  = LocalClusteringAlgorithm::NOOP,
                .local_lp =
                    {
                        .num_iterations                       = 5,
                        .passive_high_degree_threshold        = 1'000'000, // unused
                        .active_high_degree_threshold         = 1'000'000,
                        .max_num_neighbors                    = kInvalidNodeID,
                        .merge_singleton_clusters             = true,
                        .merge_nonadjacent_clusters_threshold = 0.5,
                        .total_num_chunks                     = 0, // unused
                        .num_chunks                           = 0, // unused
                        .min_num_chunks                       = 0, // unused
                        .ignore_ghost_nodes                   = false,
                        .keep_ghost_clusters                  = false,
                        .scale_chunks_with_threads            = false, // unused
                    },
                .contraction_limit         = 5000,
                .cluster_weight_limit      = shm::ClusterWeightLimit::EPSILON_BLOCK_WEIGHT,
                .cluster_weight_multiplier = 1.0,
            },
        .initial_partitioning =
            {
                .algorithm = InitialPartitioningAlgorithm::KAMINPAR,
                .mtkahypar =
                    {
                        .preset_filename = "",
                    },
                .kaminpar = shm::create_default_context(),
            },
        .refinement =
            {
                .algorithms =
                    {KWayRefinementAlgorithm::GREEDY_BALANCER, KWayRefinementAlgorithm::LP,
                     KWayRefinementAlgorithm::GREEDY_BALANCER},
                .lp =
                    {
                        .active_high_degree_threshold = 1'000'000,
                        .num_iterations               = 5,
                        .total_num_chunks             = 128,
                        .num_chunks                   = 0,
                        .min_num_chunks               = 8,
                        .num_move_attempts            = 2,
                        .ignore_probabilities         = true,
                        .scale_chunks_with_threads    = false,
                    },
                .colored_lp =
                    {
                        .num_iterations                     = 5,
                        .num_move_execution_iterations      = 1,
                        .num_probabilistic_move_attempts    = 2,
                        .sort_by_rel_gain                   = true,
                        .num_coloring_chunks                = 0,
                        .max_num_coloring_chunks            = 128,
                        .min_num_coloring_chunks            = 8,
                        .scale_coloring_chunks_with_threads = false,
                        .small_color_blacklist              = 0,
                        .only_blacklist_input_level         = false,
                        .track_local_block_weights          = true,
                        .use_active_set                     = false,
                        .move_execution_strategy            = LabelPropagationMoveExecutionStrategy::BEST_MOVES,
                    },
                .fm =
                    {
                        .alpha           = 1.0,
                        .radius          = 3,
                        .pe_radius       = 2,
                        .overlap_regions = false,
                        .num_iterations  = 5,
                        .sequential      = false,
                        .premove_locally = true,
                        .bound_degree    = 0,
                        .contract_border = false,
                    },
                .greedy_balancer =
                    {
                        .num_nodes_per_block = 5,
                    },
                .refine_coarsest_level = false,
            },
        .debug = {
            .save_finest_graph               = false,
            .save_coarsest_graph             = false,
            .save_graph_hierarchy            = false,
            .save_clustering_hierarchy       = false,
            .save_partition_hierarchy        = false,
            .save_unrefined_finest_partition = false,
        }};
}

Context create_ipdps23_submission_default_context() {
    Context ctx        = create_default_context();
    ctx.partition.mode = PartitioningMode::DEEP;
    return ctx;
}

Context create_strong_context() {
    Context ctx                             = create_default_context();
    ctx.initial_partitioning.algorithm      = InitialPartitioningAlgorithm::MTKAHYPAR;
    ctx.coarsening.global_lp.num_iterations = 5;
    return ctx;
}

Context create_ipdps23_submission_strong_context() {
    Context ctx                             = create_ipdps23_submission_default_context();
    ctx.initial_partitioning.algorithm      = InitialPartitioningAlgorithm::MTKAHYPAR;
    ctx.coarsening.global_lp.num_iterations = 5;
    return ctx;
}
} // namespace kaminpar::dist
