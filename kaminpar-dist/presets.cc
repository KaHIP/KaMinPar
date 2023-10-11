/*******************************************************************************
 * Configuration presets.
 *
 * @file:   presets.cc
 * @author: Daniel Seemaier
 * @date:   15.10.2022
 ******************************************************************************/
#include "kaminpar-dist/presets.h"

#include <stdexcept>

#include "kaminpar-dist/context.h"

#include "kaminpar-shm/presets.h"

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
  } else if (name == "jet") {
    return create_jet_context();
  } else if (name == "fm") {
    return create_fm_context();
  }

  throw std::runtime_error("invalid preset name");
}

std::unordered_set<std::string> get_preset_names() {
  return {
      "default",
      "strong",
      "europar23-fast",
      "europar23-strong",
      "jet",
      "fm",
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
                      .chunks =
                          {
                              .total_num_chunks = 128,
                              .fixed_num_chunks = 0,
                              .min_num_chunks = 8,
                              .scale_chunks_with_threads = false,
                          },
                      .keep_ghost_clusters = false,
                      .sync_cluster_weights = true,
                      .enforce_cluster_weights = true,
                      .cheap_toplevel = false,
                      .prevent_cyclic_moves = false,
                      .enforce_legacy_weight = false,
                  },
              .hem =
                  {
                      .chunks =
                          {
                              .total_num_chunks = 128,
                              .fixed_num_chunks = 0,
                              .min_num_chunks = 8,
                              .scale_chunks_with_threads = false,
                          },
                      .only_blacklist_input_level = false,
                      .ignore_weight_limit = false,
                  },
              .max_local_clustering_levels = 0,
              .local_clustering_algorithm = LocalClusteringAlgorithm::NOOP,
              .local_lp =
                  {
                      .num_iterations = 5,
                      .active_high_degree_threshold = 1'000'000,
                      .max_num_neighbors = kInvalidNodeID,
                      .merge_singleton_clusters = false,
                      .merge_nonadjacent_clusters_threshold = 0.5,
                      .ignore_ghost_nodes = true,
                      .keep_ghost_clusters = false,
                  },
              .contraction_limit = 2000,
              .cluster_weight_limit = shm::ClusterWeightLimit::EPSILON_BLOCK_WEIGHT,
              .cluster_weight_multiplier = 1.0,
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
                  {RefinementAlgorithm::GREEDY_NODE_BALANCER,
                   RefinementAlgorithm::BATCHED_LP,
                   RefinementAlgorithm::GREEDY_NODE_BALANCER},
              .refine_coarsest_level = false,
              .lp =
                  {
                      .active_high_degree_threshold = 1'000'000,
                      .num_iterations = 5,
                      .chunks =
                          {
                              .total_num_chunks = 128,
                              .fixed_num_chunks = 0,
                              .min_num_chunks = 8,
                              .scale_chunks_with_threads = false,
                          },
                      .num_move_attempts = 2,
                      .ignore_probabilities = true,
                  },
              .colored_lp =
                  {
                      .num_iterations = 5,
                      .num_move_execution_iterations = 1,
                      .num_probabilistic_move_attempts = 2,
                      .sort_by_rel_gain = true,
                      .coloring_chunks =
                          {
                              .total_num_chunks = 128,
                              .fixed_num_chunks = 0,
                              .min_num_chunks = 8,
                              .scale_chunks_with_threads = false,
                          },
                      .small_color_blacklist = 0,
                      .only_blacklist_input_level = false,
                      .track_local_block_weights = true,
                      .use_active_set = false,
                      .move_execution_strategy = LabelPropagationMoveExecutionStrategy::BEST_MOVES,
                  },
              .fm =
                  {
                      .alpha = 1.0,

                      // -- local FM --
                      .overlap_regions = false,
                      .bound_degree = 0,
                      .contract_border = false,

                      // -- mostly global FM, some local FM --
                      .use_independent_seeds = true,
                      .use_bfs_seeds_as_fm_seeds = true,

                      .chunk_local_rounds = false,
                      .chunks =
                          {
                              .total_num_chunks = 128,
                              .fixed_num_chunks = 0,
                              .min_num_chunks = 8,
                              .scale_chunks_with_threads = false,
                          },

                      .max_hops = 1,
                      .max_radius = 1,

                      .num_global_iterations = 10,
                      .num_local_iterations = 1,

                      .revert_local_moves_after_batch = true,
                      .rebalance_after_each_global_iteration = true,
                      .rebalance_after_refinement = false,
                      .balancing_algorithm = RefinementAlgorithm::GREEDY_NODE_BALANCER,

                      .rollback_deterioration = true,

                      .use_abortion_threshold = true,
                      .abortion_threshold = 0.999,
                  },
              .node_balancer =
                  {
                      .max_num_rounds = std::numeric_limits<int>::max(),
                      .enable_sequential_balancing = true,
                      .seq_num_nodes_per_block = 5,
                      .enable_parallel_balancing = true,
                      .par_threshold = 0.1,
                      .par_num_dicing_attempts = 0,
                      .par_accept_imbalanced_moves = true,
                      .par_enable_positive_gain_buckets = true,
                      .par_gain_bucket_base = 1.1,
                  },
              .cluster_balancer =
                  {
                      .max_num_rounds = std::numeric_limits<int>::max(),
                      .enable_sequential_balancing = true,
                      .seq_num_nodes_per_block = 5,
                      .seq_full_pq = true,
                      .enable_parallel_balancing = true,
                      .parallel_threshold = 0.1,
                      .par_num_dicing_attempts = 0,
                      .par_accept_imbalanced = true,
                      .par_use_positive_gain_buckets = true,
                      .par_gain_bucket_factor = 1.1,
                      .par_initial_rebalance_fraction = 1.0,
                      .par_rebalance_fraction_increase = 0.01,
                      .cluster_size_strategy = ClusterSizeStrategy::ONE,
                      .cluster_size_multiplier = 1.0,
                      .cluster_strategy = ClusterStrategy::SINGLETONS,
                      .cluster_rebuild_interval = 0,
                      .switch_to_sequential_after_stallmate = true,
                      .switch_to_singleton_after_stallmate = true,
                  },
              .jet =
                  {
                      .num_iterations = 0,
                      .num_fruitless_iterations = 12,
                      .fruitless_threshold = 0.999,
                      .coarse_penalty_factor = 0.25,
                      .fine_penalty_factor = 0.75,
                      .balancing_algorithm = RefinementAlgorithm::GREEDY_NODE_BALANCER,
                  },
              .jet_balancer =
                  {
                      .num_weak_iterations = 2,
                      .num_strong_iterations = 1,
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
      RefinementAlgorithm::GREEDY_NODE_BALANCER,
      RefinementAlgorithm::BATCHED_LP,
      RefinementAlgorithm::JET_REFINER};
  return ctx;
}

Context create_europar23_fast_context() {
  Context ctx = create_default_context();
  ctx.coarsening.global_lp.enforce_legacy_weight = true;
  return ctx;
}

Context create_europar23_strong_context() {
  Context ctx = create_europar23_fast_context();
  ctx.initial_partitioning.algorithm = InitialPartitioningAlgorithm::MTKAHYPAR;
  ctx.coarsening.global_lp.num_iterations = 5;
  return ctx;
}

Context create_jet_context() {
  Context ctx = create_default_context();
  ctx.refinement.algorithms = {
      RefinementAlgorithm::GREEDY_NODE_BALANCER,
      RefinementAlgorithm::BATCHED_LP,
      RefinementAlgorithm::JET_REFINER,
      RefinementAlgorithm::GREEDY_NODE_BALANCER};
  return ctx;
}

Context create_fm_context() {
  Context ctx = create_default_context();
  ctx.refinement.algorithms = {
      RefinementAlgorithm::GREEDY_NODE_BALANCER,
      RefinementAlgorithm::BATCHED_LP,
      RefinementAlgorithm::GLOBAL_FM,
      RefinementAlgorithm::GREEDY_NODE_BALANCER};
  return ctx;
}
} // namespace kaminpar::dist
