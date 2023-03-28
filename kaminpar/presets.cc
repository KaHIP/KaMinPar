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
  if (name == "default" || name == "fast") {
    return create_default_context();
  } else if (name == "largek") {
    return create_largek_context();
  } else if (name == "fm") {
    return create_fm_context();
  }

  throw std::runtime_error("invalid preset name");
}

std::unordered_set<std::string> get_preset_names() {
  return {
      "default",
      "fast",
      "largek",
      "fm",
  };
}

Context create_default_context() {
  return {
      // Context
      .graph_filename = "",
      .seed = 0,
      .save_partition = false,
      .partition_directory = "./",
      .partition_filename = "", // generate filename
      .degree_weights = false,
      .quiet = false,
      .parsable_output = false,
      .unchecked_io = false,
      .validate_io = false,
      .partition =
          {
              // Context -> Partition
              .mode = PartitioningMode::DEEP,
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
              // Context -> Initial Partitioning
              .coarsening =
                  {
                      // Context -> Initial Partitioning -> Coarsening
                      .algorithm = ClusteringAlgorithm::LABEL_PROPAGATION,
                      .lp =
                          {
                              // Context -> Initial Partitioning -> Coarsening
                              // -> Label Propagation
                              .num_iterations = 1,                 // no effect
                              .large_degree_threshold = 1000000,   // no effect
                              .max_num_neighbors = 200000,         // no effect
                              .two_hop_clustering_threshold = 0.5, // no effect
                          },
                      .contraction_limit = 20,
                      .enforce_contraction_limit = false, // no effect
                      .convergence_threshold = 0.05,
                      .cluster_weight_limit = ClusterWeightLimit::BLOCK_WEIGHT,
                      .cluster_weight_multiplier = 1.0 / 12.0,
                  },
              .refinement =
                  {
                      // Context -> Initial Partitioning -> Refinement
                      .algorithms = {RefinementAlgorithm::TWO_WAY_FM},
                      .lp = {},
                      .fm =
                          {
                              // Context -> Initial Partitioning -> Refinement
                              // -> FM
                              .stopping_rule = FMStoppingRule::SIMPLE,
                              .num_fruitless_moves = 100,
                              .alpha = 1.0,
                              .num_iterations = 5,
                              .improvement_abortion_threshold = 0.0001,
                          },
                      .balancer = {},
                  },
              .mode = InitialPartitioningMode::SYNCHRONOUS_PARALLEL,
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
                  {RefinementAlgorithm::GREEDY_BALANCER,
                   RefinementAlgorithm::LABEL_PROPAGATION},
              .lp =
                  {
                      // Context -> Refinement -> Label Propagation
                      .num_iterations = 5,
                      .large_degree_threshold = 1000000,
                      .max_num_neighbors = std::numeric_limits<NodeID>::max(),
                  },
              .fm = {},
              .balancer = {},
          },
      .parallel =
          {
              // Context -> Parallel
              .use_interleaved_numa_allocation = true,
              .num_threads = 1,
          },
  };
}

Context create_largek_context() {
  Context ctx = create_default_context();

  ctx.initial_partitioning.min_num_repetitions = 4;
  ctx.initial_partitioning.min_num_non_adaptive_repetitions = 2;
  ctx.initial_partitioning.max_num_repetitions = 4;

  return ctx;
}

Context create_fm_context() {
  Context ctx = create_default_context();
  ctx.refinement.algorithms = {
      RefinementAlgorithm::GREEDY_BALANCER,
      RefinementAlgorithm::LABEL_PROPAGATION,
      RefinementAlgorithm::KWAY_FM,
  };
  return ctx;
}
} // namespace kaminpar::shm
