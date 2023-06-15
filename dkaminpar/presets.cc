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
Context create_context_by_preset_name(const std::string &name) {
  if (name == "default" || name == "fast") {
    return create_default_context();
  } else if (name == "strong") {
    return create_strong_context();
  } else if (name == "europar23-fast") {
    return create_europar23_fast_context();
  } else if (name == "europar23-strong") {
    return create_europar23_strong_context();
  }

  throw std::runtime_error("invalid preset name");
}

std::unordered_set<std::string> get_preset_names() {
  return {
      "default",
      "strong",
      "europar23-fast",
      "europar23-strong",
  };
}

Context create_default_context() {
  return {
      .rearrange_by = GraphOrdering::DEGREE_BUCKETS,
      .mode = PartitioningMode::DEEP,
      .enable_pe_splitting = true,
      .simulate_singlethread = true,
      .partition =
          {
              kInvalidBlockID, // k
              128,             // K
              0.03,            // epsilon
          },
      .parallel =
          {
              .num_threads = 1,
              .num_mpis = 1,
          },
      .coarsening =
          {
              .max_global_clustering_levels = std::numeric_limits<std::size_t>::max(),
              .global_clustering_algorithm = GlobalClusteringAlgorithm::LP,
              .global_lp =
                  {
                      .num_iterations = 3,
                      .passive_high_degree_threshold = 1'000'000,
                      .active_high_degree_threshold = 1'000'000,
                      .max_num_neighbors = kInvalidNodeID,
                      .merge_singleton_clusters = true,
                      .merge_nonadjacent_clusters_threshold = 0.5,
                      .total_num_chunks = 128,
                      .fixed_num_chunks = 0,
                      .min_num_chunks = 8,
                      .ignore_ghost_nodes = false, // unused
                      .keep_ghost_clusters = false,
                      .scale_chunks_with_threads = false,
                      .sync_cluster_weights = true,
                      .enforce_cluster_weights = true,
                      .cheap_toplevel = false,
                      .prevent_cyclic_moves = false,
                      .enforce_legacy_weight = false,
                  },
              .hem =
                  {
                      .max_num_coloring_chunks = 128,
                      .fixed_num_coloring_chunks = 0,
                      .min_num_coloring_chunks = 8,
                      .scale_coloring_chunks_with_threads = false,
                      .small_color_blacklist = 0,
                      .only_blacklist_input_level = false,
                      .ignore_weight_limit = false,
                  },
              .max_local_clustering_levels = 0,
              .local_clustering_algorithm = LocalClusteringAlgorithm::NOOP,
              .local_lp =
                  {
                      .num_iterations = 5,
                      .passive_high_degree_threshold = 1'000'000, // unused
                      .active_high_degree_threshold = 1'000'000,
                      .max_num_neighbors = kInvalidNodeID,
                      .merge_singleton_clusters = true,
                      .merge_nonadjacent_clusters_threshold = 0.5,
                      .total_num_chunks = 0, // unused
                      .fixed_num_chunks = 0, // unused
                      .min_num_chunks = 0,   // unused
                      .ignore_ghost_nodes = false,
                      .keep_ghost_clusters = false,
                      .scale_chunks_with_threads = false, // unused
                  },
              .contraction_limit = 2000,
              .cluster_weight_limit = shm::ClusterWeightLimit::EPSILON_BLOCK_WEIGHT,
              .cluster_weight_multiplier = 1.0,
              .contraction_algorithm = ContractionAlgorithm::DEFAULT,
              .max_cnode_imbalance = 1.1,
              .migrate_cnode_prefix = true,
              .force_perfect_cnode_balance = false,
          },
      .initial_partitioning =
          {
              .algorithm = InitialPartitioningAlgorithm::KAMINPAR,
              .kaminpar = shm::create_default_context(),
          },
      .refinement =
          {
              .algorithms =
                  {KWayRefinementAlgorithm::GREEDY_BALANCER,
                   KWayRefinementAlgorithm::LP,
                   KWayRefinementAlgorithm::GREEDY_BALANCER},
              .refine_coarsest_level = false,
              .lp =
                  {
                      .active_high_degree_threshold = 1'000'000,
                      .num_iterations = 5,
                      .total_num_chunks = 128,
                      .fixed_num_chunks = 0,
                      .min_num_chunks = 8,
                      .num_move_attempts = 2,
                      .ignore_probabilities = true,
                      .scale_chunks_with_threads = false,
                  },
              .colored_lp =
                  {
                      .num_iterations = 5,
                      .num_move_execution_iterations = 1,
                      .num_probabilistic_move_attempts = 2,
                      .sort_by_rel_gain = true,
                      .max_num_coloring_chunks = 128,
                      .fixed_num_coloring_chunks = 0,
                      .min_num_coloring_chunks = 8,
                      .scale_coloring_chunks_with_threads = false,
                      .small_color_blacklist = 0,
                      .only_blacklist_input_level = false,
                      .track_local_block_weights = true,
                      .use_active_set = false,
                      .move_execution_strategy = LabelPropagationMoveExecutionStrategy::BEST_MOVES,
                  },
              .fm =
                  {
                      .alpha = 1.0,
                      .radius = 3,
                      .pe_radius = 2,
                      .overlap_regions = false,
                      .num_iterations = 5,
                      .sequential = false,
                      .premove_locally = true,
                      .bound_degree = 0,
                      .contract_border = false,
                  },
              .greedy_balancer =
                  {
                      .max_num_rounds = std::numeric_limits<int>::max(),
                      .enable_strong_balancing = true,
                      .num_nodes_per_block = 5,
                      .enable_fast_balancing = false,
                      .fast_balancing_threshold = 0.1,
                  },
              .jet =
                  {
                      .num_iterations = 12,
                      .min_c = 0.25,
                      .max_c = 0.75,
                      .interpolate_c = false,
                      .use_abortion_threshold = true,
                      .abortion_threshold = 0.999,
                  },
          },
      .debug = {
          .save_coarsest_graph = false,
          .save_coarsest_partition = false,
      }};
}

Context create_strong_context() {
  Context ctx = create_default_context();
  ctx.initial_partitioning.kaminpar = shm::create_strong_context();
  ctx.coarsening.global_lp.num_iterations = 5;
  ctx.refinement.algorithms = {
      KWayRefinementAlgorithm::GREEDY_BALANCER,
      KWayRefinementAlgorithm::LP,
      KWayRefinementAlgorithm::JET};
  return ctx;
}

Context create_europar23_fast_context() {
  return create_default_context();
}

Context create_europar23_strong_context() {
  Context ctx = create_default_context();
  ctx.initial_partitioning.algorithm = InitialPartitioningAlgorithm::MTKAHYPAR;
  ctx.coarsening.global_lp.num_iterations = 5;
  return ctx;
}
} // namespace kaminpar::dist
