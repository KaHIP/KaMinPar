/*******************************************************************************
 * Configuration presets for KaMinPar.
 *
 * @file:   presets.cc
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 ******************************************************************************/
#include "kaminpar-shm/presets.h"

#include <stdexcept>
#include <string>
#include <unordered_set>

#include "kaminpar-shm/context.h"

namespace kaminpar::shm {
Context create_context_by_preset_name(const std::string &name) {
  if (name == "default") {
    return create_default_context();
  } else if (name == "fast") {
    return create_fast_context();
  } else if (name == "largek") {
    return create_largek_context();
  } else if (name == "strong" || name == "fm") {
    return create_strong_context();
  } else if (name == "jet") {
    return create_jet_context();
  } else if (name == "noref") {
    return create_noref_context();
  }

  throw std::runtime_error("invalid preset name");
}

std::unordered_set<std::string> get_preset_names() {
  return {
      "default",
      "fast",
      "largek",
      "strong",
      "fm",
      "jet",
      "noref",
  };
}

Context create_default_context() {
  return {
      .rearrange_by = GraphOrdering::DEGREE_BUCKETS,
      .partitioning =
          {
              .mode = PartitioningMode::DEEP,
              .deep_initial_partitioning_mode = InitialPartitioningMode::ASYNCHRONOUS_PARALLEL,
              .deep_initial_partitioning_load = 1.0,
          },
      .partition =
          {
              // Context -> Partition
              .epsilon = 0.03,
              .k = kInvalidBlockID /* must be set */,
          },
      .coarsening =
          {
              // Context -> Coarsening
              .algorithm = ClusteringAlgorithm::LABEL_PROPAGATION,
              .lp =
                  {
                      // Context -> Coarsening -> Label Propagation
                      .num_iterations = 5,
                      .large_degree_threshold = 1000000,
                      .max_num_neighbors = 200000,
                      .two_hop_strategy = TwoHopStrategy::MATCH_THREADWISE,
                      .two_hop_threshold = 0.5,
                      .isolated_nodes_strategy =
                          IsolatedNodesClusteringStrategy::MATCH_DURING_TWO_HOP,
                  },
              .contraction_limit = 2000,
              .enforce_contraction_limit = false,
              .convergence_threshold = 0.05,
              .cluster_weight_limit = ClusterWeightLimit::EPSILON_BLOCK_WEIGHT,
              .cluster_weight_multiplier = 1.0,
          },
      .initial_partitioning =
          {
              .coarsening =
                  {
                      .contraction_limit = 20,
                      .convergence_threshold = 0.05,
                      .large_degree_threshold = 1000000,
                      .cluster_weight_limit = ClusterWeightLimit::BLOCK_WEIGHT,
                      .cluster_weight_multiplier = 1.0 / 12.0,
                  },
              .refinement =
                  {
                      .disabled = false,
                      .stopping_rule = FMStoppingRule::SIMPLE,
                      .num_fruitless_moves = 100,
                      .alpha = 1.0,
                      .num_iterations = 5,
                      .improvement_abortion_threshold = 0.0001,
                  },
              .repetition_multiplier = 1.0,
              .min_num_repetitions = 10,
              .min_num_non_adaptive_repetitions = 5,
              .max_num_repetitions = 50,
              .num_seed_iterations = 1,
              .use_adaptive_bipartitioner_selection = true,
          },
      .refinement =
          {
              // Context -> Refinement
              .algorithms =
                  {RefinementAlgorithm::GREEDY_BALANCER, RefinementAlgorithm::LABEL_PROPAGATION},
              .lp =
                  {
                      // Context -> Refinement -> Label Propagation
                      .num_iterations = 5,
                      .large_degree_threshold = 1000000,
                      .max_num_neighbors = std::numeric_limits<NodeID>::max(),
                  },
              .kway_fm =
                  {
                      .num_seed_nodes = 10,
                      .alpha = 1.0,
                      .num_iterations = 10,
                      .unlock_locally_moved_nodes = true,
                      .unlock_seed_nodes = true,
                      .use_exact_abortion_threshold = false,
                      .abortion_threshold = 0.999,
                      .gain_cache_strategy = GainCacheStrategy::DENSE,
                      .constant_high_degree_threshold = 0,
                      .k_based_high_degree_threshold = 1.0,

                      .dbg_compute_batch_stats = false,
                  },
              .balancer = {},
              .jet =
                  {
                      .num_iterations = 0,
                      .num_fruitless_iterations = 12,
                      .fruitless_threshold = 0.999,
                      .fine_negative_gain_factor = 0.25,
                      .coarse_negative_gain_factor = 0.75,
                      .balancing_algorithm = RefinementAlgorithm::GREEDY_BALANCER,
                  },
              .mtkahypar =
                  {
                      .config_filename = "",
                      .coarse_config_filename = "",
                      .fine_config_filename = "",
                  },
          },
      .parallel =
          {
              // Context -> Parallel
              .num_threads = 1,
          },
      .debug =
          {
              .graph_name = "",

              .dump_graph_filename = "n%n_m%m_k%k_seed%seed.metis",
              .dump_partition_filename = "n%n_m%m_k%k_seed%seed.part",

              .dump_toplevel_graph = false,
              .dump_toplevel_partition = false,
              .dump_coarsest_graph = false,
              .dump_coarsest_partition = false,
              .dump_graph_hierarchy = false,
              .dump_partition_hierarchy = false,
          },
  };
}

Context create_fast_context() {
  Context ctx = create_default_context();
  ctx.partitioning.deep_initial_partitioning_mode = InitialPartitioningMode::ASYNCHRONOUS_PARALLEL;
  ctx.partitioning.deep_initial_partitioning_load = 0.5;
  ctx.coarsening.lp.num_iterations = 1;
  ctx.initial_partitioning.min_num_repetitions = 1;
  ctx.initial_partitioning.min_num_non_adaptive_repetitions = 1;
  ctx.initial_partitioning.max_num_repetitions = 1;
  return ctx;
}

Context create_largek_context() {
  Context ctx = create_default_context();

  ctx.initial_partitioning.min_num_repetitions = 4;
  ctx.initial_partitioning.min_num_non_adaptive_repetitions = 2;
  ctx.initial_partitioning.max_num_repetitions = 4;

  return ctx;
}

Context create_strong_context() {
  Context ctx = create_default_context();

  ctx.refinement.algorithms = {
      RefinementAlgorithm::GREEDY_BALANCER,
      RefinementAlgorithm::LABEL_PROPAGATION,
      RefinementAlgorithm::KWAY_FM,
      RefinementAlgorithm::GREEDY_BALANCER,
  };

  return ctx;
}

Context create_jet_context() {
  Context ctx = create_default_context();
  ctx.refinement.algorithms = {
      RefinementAlgorithm::GREEDY_BALANCER,
      RefinementAlgorithm::JET,
  };
  return ctx;
}

Context create_noref_context() {
  Context ctx = create_default_context();
  ctx.refinement.algorithms.clear();
  return ctx;
}
} // namespace kaminpar::shm
