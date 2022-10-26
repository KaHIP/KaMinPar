/*******************************************************************************
 * @file:   presets.cc
 * @author: Daniel Seemaier
 * @date:   15.10.2022
 * @brief:  Configuration presets.
 ******************************************************************************/
#include "dkaminpar/presets.h"

#include "kaminpar/presets.h"

namespace kaminpar::dist {
Context create_default_context() {
    return {
        .graph_filename     = "",
        .load_edge_balanced = false,
        .seed               = 0,
        .quiet              = false,
        .num_repetitions    = 0,
        .time_limit         = 0,
        .sort_graph         = true,
        .parsable_output    = false,
        .partition          = {
                     /* .k = */ 0,
            /* .k_prime = */ 128,
            /* .epsilon = */ 0.03,
            /* .mode = */ PartitioningMode::DEEPER,
            /* .enable_pe_splitting = */ false,
        },
        .parallel =
            {
                .num_threads                     = 1,
                .num_mpis                        = 1,
                .use_interleaved_numa_allocation = true,
                .mpi_thread_support              = MPI_THREAD_FUNNELED,
                .simulate_singlethread           = true,
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
                .algorithm = KWayRefinementAlgorithm::LP,
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
                .balancing =
                    {
                        .algorithm           = BalancingAlgorithm::DISTRIBUTED,
                        .num_nodes_per_block = 5,
                    },
                .refine_coarsest_level = false,
            },
        .debug = {
            .save_imbalanced_partitions = false,
            .save_graph_hierarchy       = false,
            .save_coarsest_graph        = false,
            .save_clustering_hierarchy  = false,
        }};
}

Context create_strong_context() {
    Context ctx                             = create_default_context();
    ctx.initial_partitioning.algorithm      = InitialPartitioningAlgorithm::MTKAHYPAR;
    ctx.coarsening.global_lp.num_iterations = 5;
    return ctx;
}
} // namespace kaminpar::dist
