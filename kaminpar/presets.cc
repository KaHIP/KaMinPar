/*******************************************************************************
 * @file:   presets.cc
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 * @brief:  Configuration presets for shared-memory partitioning.
 ******************************************************************************/
#include "kaminpar/presets.h"

#include <stdexcept>
#include <string>
#include <unordered_set>

#include "kaminpar/context.h"

namespace kaminpar::shm {
Context create_context_by_preset_name(const std::string &name) {
  if (name == "default") {
    return create_default_context();
  } else if (name == "fast") {
    return create_fast_context();
  } else if (name == "largek") {
    return create_largek_context();
  } else if (name == "strong") {
    return create_strong_context();
  } else if (name == "jet") {
    return create_jet_context();
  }

  throw std::runtime_error("invalid preset name");
}

std::unordered_set<std::string> get_preset_names() {
  return {
      "default",
      "fast",
      "largek",
      "strong",
      "jet",
  };
}

Context create_default_context() {
  return {
      .mode = PartitioningMode::DEEP,
      // Context
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
                      .two_hop_clustering_threshold = 0.5,
                  },
              .contraction_limit = 2000,
              .enforce_contraction_limit = false,
              .convergence_threshold = 0.05,
              .cluster_weight_limit = ClusterWeightLimit::EPSILON_BLOCK_WEIGHT,
              .cluster_weight_multiplier = 1.0,
          },
      .initial_partitioning =
          {
              .mode = InitialPartitioningMode::SYNCHRONOUS_PARALLEL,
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
              .multiplier_exponent = 0,
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
                      .unlock_seed_nodes = true,
                      .use_exact_abortion_threshold = false,
                      .abortion_threshold = 0.999,
                  },
              .balancer = {},
              .jet =
                  {
                      .num_iterations = 12,
                      .interpolate_c = false,
                      .min_c = 0.25,
                      .max_c = 0.75,
                      .abortion_threshold = 0.999,
                  },
              .mtkahypar =
                  {
                      .config_filename = "",
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
              .dump_coarsest_graph = false,
              .dump_coarsest_partition = false,
              .dump_graph_hierarchy = false,
              .dump_partition_hierarchy = false,
          },
  };
}

Context create_fast_context() {
  Context ctx = create_default_context();
  ctx.coarsening.lp.num_iterations = 1;
  ctx.initial_partitioning.min_num_repetitions = 1;
  ctx.initial_partitioning.min_num_non_adaptive_repetitions = 1;
  ctx.initial_partitioning.max_num_repetitions = 1;
  ctx.initial_partitioning.mode = InitialPartitioningMode::SEQUENTIAL;
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
  // ctx.coarsening.cluster_weight_limit = ClusterWeightLimit::BLOCK_WEIGHT;
  // ctx.coarsening.cluster_weight_multiplier = 1.0 / 18.0;

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
} // namespace kaminpar::shm
