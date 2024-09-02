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
  } else if (name == "strong" || name == "fm") {
    return create_strong_context();
  }

  if (name == "largek") {
    return create_largek_context();
  } else if (name == "fast-largek") {
    return create_fast_largek_context();
  } else if (name == "strong-largek") {
    return create_strong_largek_context();
  }

  if (name == "memory") {
    return create_memory_context();
  } else if (name == "strong-memory") {
    return create_strong_memory_context();
  }

  if (name == "jet") {
    return create_jet_context(1);
  } else if (name == "4xjet") {
    return create_jet_context(4);
  } else if (name == "noref") {
    return create_noref_context();
  }

  throw std::runtime_error("invalid preset name");
}

std::unordered_set<std::string> get_preset_names() {
  return {
      "default",
      "fast",
      "strong",
      "largek",
      "fast-largek",
      "strong-largek",
      "memory",
      "strong-memory",
      "jet",
      "4xjet",
      "noref",
      "fm",
  };
}

Context create_default_context() {
  return {
      .compression =
          {
              .enabled = false,
          },
      .node_ordering = NodeOrdering::DEGREE_BUCKETS,
      .edge_ordering = EdgeOrdering::NATURAL,
      .partitioning =
          {
              .mode = PartitioningMode::DEEP,
              .deep_initial_partitioning_mode = InitialPartitioningMode::ASYNCHRONOUS_PARALLEL,
              .deep_initial_partitioning_load = 1.0,
              .min_consecutive_seq_bipartitioning_levels = 1,
              .use_lazy_subgraph_memory = false,
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
              .algorithm = CoarseningAlgorithm::CLUSTERING,
              .clustering =
                  {
                      // Context -> Coarsening -> Clustering
                      .algorithm = ClusteringAlgorithm::LABEL_PROPAGATION,
                      .lp =
                          {
                              // Context -> Coarsening -> Clustering -> Label Propagation
                              .num_iterations = 5,
                              .large_degree_threshold = 1000000,
                              .max_num_neighbors = 200000,
                              .tie_breaking_strategy = TieBreakingStrategy::UNIFORM,
                              .cluster_weights_structure = ClusterWeightsStructure::VEC,
                              .impl = LabelPropagationImplementation::TWO_PHASE,
                              .second_phase_selection_strategy =
                                  SecondPhaseSelectionStrategy::FULL_RATING_MAP,
                              .second_phase_aggregation_strategy =
                                  SecondPhaseAggregationStrategy::BUFFERED,
                              .relabel_before_second_phase = false,
                              .two_hop_strategy = TwoHopStrategy::MATCH_THREADWISE,
                              .two_hop_threshold = 0.5,
                              .isolated_nodes_strategy =
                                  IsolatedNodesClusteringStrategy::MATCH_DURING_TWO_HOP,
                          },
                      .cluster_weight_limit = ClusterWeightLimit::EPSILON_BLOCK_WEIGHT,
                      .cluster_weight_multiplier = 1.0,
                      .max_mem_free_coarsening_level = 0,
                  },
              .contraction =
                  {
                      // Context -> Coarsening -> Contraction
                      .mode = ContractionMode::BUFFERED,
                      .implementation = ContractionImplementation::SINGLE_PHASE,
                      .edge_buffer_fill_fraction = 1,
                  },
              .contraction_limit = 2000,
              .convergence_threshold = 0.05,
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
              .pool =
                  {
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
                      .enable_bfs_bipartitioner = true,
                      .enable_ggg_bipartitioner = true,
                      .enable_random_bipartitioner = true,
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
              .refine_pool_partition = false,
          },
      .refinement =
          {
              // Context -> Refinement
              .algorithms =
                  {
                      RefinementAlgorithm::GREEDY_BALANCER,
                      RefinementAlgorithm::LEGACY_LABEL_PROPAGATION,
                  },
              .lp =
                  {
                      // Context -> Refinement -> Label Propagation
                      .num_iterations = 5,
                      .large_degree_threshold = 1000000,
                      .max_num_neighbors = std::numeric_limits<NodeID>::max(),
                      .impl = LabelPropagationImplementation::SINGLE_PHASE,
                      .tie_breaking_strategy = TieBreakingStrategy::UNIFORM,
                      .second_phase_selection_strategy =
                          SecondPhaseSelectionStrategy::FULL_RATING_MAP,
                      .second_phase_aggregation_strategy = SecondPhaseAggregationStrategy::BUFFERED,
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
                      .gain_cache_strategy = GainCacheStrategy::COMPACT_HASHING,
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
                      .num_rounds_on_fine_level = 1,
                      .num_rounds_on_coarse_level = 1,
                      .initial_gain_temp_on_fine_level = 0.25,
                      .final_gain_temp_on_fine_level = 0.25,
                      .initial_gain_temp_on_coarse_level = 0.75,
                      .final_gain_temp_on_coarse_level = 0.75,
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
  ctx.partitioning.deep_initial_partitioning_load = 0.5;
  ctx.coarsening.clustering.lp.num_iterations = 1;
  ctx.initial_partitioning.pool.min_num_repetitions = 1;
  ctx.initial_partitioning.pool.min_num_non_adaptive_repetitions = 1;
  ctx.initial_partitioning.pool.max_num_repetitions = 1;
  return ctx;
}

Context create_strong_context() {
  Context ctx = create_default_context();

  ctx.refinement.algorithms = {
      RefinementAlgorithm::GREEDY_BALANCER,
      RefinementAlgorithm::LEGACY_LABEL_PROPAGATION,
      RefinementAlgorithm::KWAY_FM,
      RefinementAlgorithm::GREEDY_BALANCER,
  };

  return ctx;
}

Context create_largek_context() {
  Context ctx = create_default_context();

  ctx.initial_partitioning.pool.min_num_repetitions = 4;
  ctx.initial_partitioning.pool.min_num_non_adaptive_repetitions = 2;
  ctx.initial_partitioning.pool.max_num_repetitions = 4;

  return ctx;
}

Context create_fast_largek_context() {
  Context ctx = create_largek_context();

  ctx.initial_partitioning.pool.min_num_repetitions = 2;
  ctx.initial_partitioning.pool.min_num_non_adaptive_repetitions = 1;
  ctx.initial_partitioning.pool.max_num_repetitions = 2;
  ctx.initial_partitioning.pool.enable_bfs_bipartitioner = true;
  ctx.initial_partitioning.pool.enable_ggg_bipartitioner = false;
  ctx.initial_partitioning.pool.enable_random_bipartitioner = true;

  ctx.initial_partitioning.pool.refinement.disabled = true;
  ctx.initial_partitioning.pool.refinement.num_iterations = 1;

  ctx.initial_partitioning.refine_pool_partition = true;

  return ctx;
}

Context create_strong_largek_context() {
  Context ctx = create_largek_context();

  ctx.refinement.algorithms = {
      RefinementAlgorithm::GREEDY_BALANCER,
      RefinementAlgorithm::LEGACY_LABEL_PROPAGATION,
      RefinementAlgorithm::KWAY_FM,
      RefinementAlgorithm::GREEDY_BALANCER,
  };

  ctx.refinement.kway_fm.gain_cache_strategy = GainCacheStrategy::COMPACT_HASHING_LARGE_K;

  return ctx;
}


Context create_memory_context() {
  Context ctx = create_default_context();
  ctx.node_ordering = NodeOrdering::EXTERNAL_DEGREE_BUCKETS;
  ctx.compression.enabled = true;
  ctx.partitioning.deep_initial_partitioning_mode = InitialPartitioningMode::SEQUENTIAL;
  ctx.partitioning.use_lazy_subgraph_memory = true;
  ctx.coarsening.clustering.max_mem_free_coarsening_level = 1;
  ctx.coarsening.clustering.lp.impl = LabelPropagationImplementation::TWO_PHASE;
  ctx.coarsening.contraction.mode = ContractionMode::UNBUFFERED;
  ctx.coarsening.contraction.implementation = ContractionImplementation::TWO_PHASE;
  ctx.refinement.kway_fm.gain_cache_strategy = GainCacheStrategy::SPARSE;

  ctx.refinement.algorithms = {
      RefinementAlgorithm::GREEDY_BALANCER,
      RefinementAlgorithm::LABEL_PROPAGATION,
  };

  return ctx;
}

Context create_strong_memory_context() {
  Context ctx = create_memory_context();

  ctx.refinement.algorithms = {
      RefinementAlgorithm::GREEDY_BALANCER,
      RefinementAlgorithm::LABEL_PROPAGATION,
      RefinementAlgorithm::KWAY_FM,
      RefinementAlgorithm::GREEDY_BALANCER,
  };

  return ctx;
}

Context create_jet_context(const int rounds) {
  Context ctx = create_default_context();
  ctx.refinement.algorithms = {
      RefinementAlgorithm::GREEDY_BALANCER,
      RefinementAlgorithm::JET,
  };

  if (rounds > 1) {
    ctx.refinement.jet.num_rounds_on_coarse_level = rounds;
    ctx.refinement.jet.num_rounds_on_fine_level = rounds;
    ctx.refinement.jet.initial_gain_temp_on_coarse_level = 0.75;
    ctx.refinement.jet.initial_gain_temp_on_fine_level = 0.75;
    ctx.refinement.jet.final_gain_temp_on_coarse_level = 0.25;
    ctx.refinement.jet.final_gain_temp_on_fine_level = 0.25;
  }

  return ctx;
}

Context create_noref_context() {
  Context ctx = create_default_context();
  ctx.refinement.algorithms.clear();
  return ctx;
}

} // namespace kaminpar::shm
