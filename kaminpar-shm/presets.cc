/*******************************************************************************
 * Configuration presets for KaMinPar.
 *
 * @file:   presets.cc
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 ******************************************************************************/
#include "kaminpar-shm/presets.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "kaminpar-shm/kaminpar.h"

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
  } else if (name == "largek-fast") {
    return create_largek_fast_context();
  } else if (name == "largek-strong") {
    return create_largek_strong_context();
  }

  if (name == "terapart") {
    return create_terapart_context();
  } else if (name == "terapart-strong") {
    return create_terapart_strong_context();
  } else if (name == "terapart-largek") {
    return create_terapart_largek_context();
  }

  if (name == "jet") {
    return create_jet_context(1);
  } else if (name == "4xjet") {
    return create_jet_context(4);
  } else if (name == "noref") {
    return create_noref_context();
  }

  if (name == "vcycle") {
    return create_vcycle_context(false);
  } else if (name == "restricted-vcycle") {
    return create_vcycle_context(true);
  }

  if (name == "esa21" || name == "esa21-smallk" || name == "diss" || name == "diss-smallk") {
    return create_esa21_smallk_context();
  } else if (name == "esa21-largek" || name == "diss-largek") {
    return create_esa21_largek_context();
  } else if (name == "esa21-largek-fast" || name == "diss-largek-fast") {
    return create_esa21_largek_fast_context();
  } else if (name == "esa21-strong" || name == "diss-strong") {
    return create_esa21_strong_context();
  }

  if (name == "mtkahypar-kway") {
    return create_mtkahypar_kway_context();
  } else if (name == "linear-time-kway") {
    return create_linear_time_kway_context();
  }

  throw std::runtime_error("invalid preset name");
}

std::unordered_set<std::string> get_preset_names() {
  return {
      "default",
      "fast",
      "strong",
      "largek",
      "terapart",
      "terapart-strong",
      "terapart-largek",
      "largek-fast",
      "largek-strong",
      "jet",
      "4xjet",
      "noref",
      "fm",
      "vcycle",
      "restricted-vcycle",
      "esa21-smallk",
      "esa21-largek",
      "esa21-largek-fast",
      "esa21-strong",
      "mtkahypar-kway",
      "linear-time-kway",
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
              .deep_initial_partitioning_mode = DeepInitialPartitioningMode::ASYNCHRONOUS_PARALLEL,
              .deep_initial_partitioning_load = 1.0,
              .min_consecutive_seq_bipartitioning_levels = 1,
              .refine_after_extending_partition = false,
              .use_lazy_subgraph_memory = true,
              .vcycles = {},
              .restrict_vcycle_refinement = false,
              .rb_enable_kway_toplevel_refinement = false,
              .rb_switch_to_seq_factor = 1,
              .kway_initial_partitioning_mode = KwayInitialPartitioningMode::PARALLEL,
          },
      .partition = {},
      .coarsening =
          {
              // Context -> Coarsening
              .algorithm = CoarseningAlgorithm::BASIC_CLUSTERING,
              .clustering =
                  {
                      // Context -> Coarsening -> Clustering
                      .algorithm = ClusteringAlgorithm::LABEL_PROPAGATION,
                      .lp =
                          {
                              // Context -> Coarsening -> Clustering -> Label Propagation
                              .num_iterations = 5,
                              .large_degree_threshold = std::numeric_limits<NodeID>::max(),
                              .max_num_neighbors = std::numeric_limits<NodeID>::max(),
                              .impl = LabelPropagationImplementation::TWO_PHASE,
                              .relabel_before_second_phase = false,
                              .two_hop_strategy = TwoHopStrategy::MATCH_THREADWISE,
                              .two_hop_threshold = 0.5,
                              .isolated_nodes_strategy =
                                  IsolatedNodesClusteringStrategy::MATCH_DURING_TWO_HOP,
                              .tie_breaking_strategy = TieBreakingStrategy::UNIFORM,
                          },

                      .cluster_weight_limit = ClusterWeightLimit::EPSILON_BLOCK_WEIGHT,
                      .cluster_weight_multiplier = 1.0,

                      .shrink_factor = std::numeric_limits<double>::max(),

                      .max_mem_free_coarsening_level = 1,

                      .forced_kc_level = false,
                      .forced_pc_level = false,
                      .forced_level_upper_factor = 10.0,
                      .forced_level_lower_factor = 1.1,
                  },
              .overlay_clustering =
                  {
                      .num_levels = 1,
                      .max_level = std::numeric_limits<int>::max(),
                  },
              .sparsification_clustering =
                  {
                      .density_target_factor = 0.5,
                      .edge_target_factor = 0.5,
                      .laziness_factor = 4,
                  },
              .contraction =
                  {
                      // Context -> Coarsening -> Contraction
                      .algorithm = ContractionAlgorithm::UNBUFFERED,
                      .unbuffered_implementation = ContractionImplementation::TWO_PHASE,
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
                              .algorithms =
                                  {
                                      InitialRefinementAlgorithm::TWOWAY_SIMPLE_FM,
                                  },
                              .fm =
                                  {
                                      .num_fruitless_moves = 100,
                                      .alpha = 1.0,
                                      .num_iterations = 5,
                                      .improvement_abortion_threshold = 0.0001,
                                  },
                              .twoway_flow =
                                  {
                                      .border_region_scaling_factor = 16,
                                      .max_border_distance = 2,
                                      .flow_algorithm = FlowAlgorithm::FIFO_PREFLOW_PUSH,
                                      .fifo_preflow_push =
                                          {

                                              .global_relabeling_heuristic = true,
                                              .global_relabeling_frequency = 1,
                                          },
                                      .highest_level_preflow_push =
                                          {
                                              .two_phase = true,
                                              .gap_heuristic = true,
                                              .global_relabeling_heuristic = true,
                                              .global_relabeling_frequency = 1,
                                          },
                                      .piercing =
                                          {
                                              .pierce_all_viable = true,
                                          },
                                      .unconstrained = false,
                                      .use_whfc = false,
                                      .parallel_scheduling = false,
                                      .max_num_rounds = std::numeric_limits<std::size_t>::max(),
                                      .min_round_improvement_factor = 0.01,
                                  },
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
                      .algorithms =
                          {
                              InitialRefinementAlgorithm::TWOWAY_SIMPLE_FM,
                          },
                      .fm =
                          {
                              .num_fruitless_moves = 100,
                              .alpha = 1.0,
                              .num_iterations = 5,
                              .improvement_abortion_threshold = 0.0001,
                          },
                      .twoway_flow =
                          {
                              .border_region_scaling_factor = 16,
                              .max_border_distance = 2,
                              .flow_algorithm = FlowAlgorithm::FIFO_PREFLOW_PUSH,
                              .fifo_preflow_push =
                                  {

                                      .global_relabeling_heuristic = true,
                                      .global_relabeling_frequency = 1,
                                  },
                              .highest_level_preflow_push =
                                  {
                                      .two_phase = true,
                                      .gap_heuristic = true,
                                      .global_relabeling_heuristic = true,
                                      .global_relabeling_frequency = 1,
                                  },
                              .piercing =
                                  {
                                      .pierce_all_viable = true,
                                  },
                              .unconstrained = false,
                              .use_whfc = false,
                              .parallel_scheduling = false,
                              .max_num_rounds = std::numeric_limits<std::size_t>::max(),
                              .min_round_improvement_factor = 0.01,
                          },
                  },
              .refine_pool_partition = false,
              .use_adaptive_epsilon = true,
          },
      .refinement =
          {
              // Context -> Refinement
              .algorithms =
                  {
                      RefinementAlgorithm::GREEDY_BALANCER,
                      RefinementAlgorithm::LABEL_PROPAGATION,
                  },
              .balancer = {},
              .lp =
                  {
                      // Context -> Refinement -> Label Propagation
                      .num_iterations = 5,
                      .large_degree_threshold = std::numeric_limits<NodeID>::max(),
                      .max_num_neighbors = std::numeric_limits<NodeID>::max(),
                      .impl = LabelPropagationImplementation::SINGLE_PHASE,
                      .tie_breaking_strategy = TieBreakingStrategy::UNIFORM,
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

                      .minimal_parallelism = std::numeric_limits<int>::max(),

                      .dbg_compute_batch_stats = false,
                      .dbg_report_progress = false,
                  },
              .twoway_flow =
                  {
                      .border_region_scaling_factor = 16,
                      .max_border_distance = 2,
                      .flow_algorithm = FlowAlgorithm::FIFO_PREFLOW_PUSH,
                      .fifo_preflow_push =
                          {

                              .global_relabeling_heuristic = true,
                              .global_relabeling_frequency = 1,
                          },
                      .highest_level_preflow_push =
                          {
                              .two_phase = true,
                              .gap_heuristic = true,
                              .global_relabeling_heuristic = true,
                              .global_relabeling_frequency = 1,
                          },
                      .piercing =
                          {
                              .pierce_all_viable = true,
                          },
                      .unconstrained = false,
                      .use_whfc = false,
                      .parallel_scheduling = true,
                      .max_num_rounds = std::numeric_limits<std::size_t>::max(),
                      .min_round_improvement_factor = 0.01,
                  },
              .multiway_flow =
                  {
                      .border_region_scaling_factor = 16,
                      .max_border_distance = 2,
                      .cut_algorithm = CutAlgorithm::ISOLATING_CUT_HEURISTIC,
                      .isolating_cut_heuristic =
                          {
                              .flow_algorithm = FlowAlgorithm::FIFO_PREFLOW_PUSH,
                              .fifo_preflow_push =
                                  {

                                      .global_relabeling_heuristic = true,
                                      .global_relabeling_frequency = 1,
                                  },
                              .highest_level_preflow_push =
                                  {
                                      .two_phase = true,
                                      .gap_heuristic = true,
                                      .global_relabeling_heuristic = true,
                                      .global_relabeling_frequency = 1,
                                  },
                          },
                      .labelling_function_heuristic =
                          {
                              .initialization_strategy =
                                  LabellingFunctionInitializationStrategy::ZERO,
                              .flow_algorithm = FlowAlgorithm::FIFO_PREFLOW_PUSH,
                              .fifo_preflow_push =
                                  {

                                      .global_relabeling_heuristic = true,
                                      .global_relabeling_frequency = 1,
                                  },
                              .highest_level_preflow_push =
                                  {
                                      .two_phase = true,
                                      .gap_heuristic = true,
                                      .global_relabeling_heuristic = true,
                                      .global_relabeling_frequency = 1,
                                  },
                              .epsilon = 0.01,
                              .max_num_rounds = std::numeric_limits<std::size_t>::max(),
                          },
                  },
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
      RefinementAlgorithm::LABEL_PROPAGATION,
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

Context create_largek_fast_context() {
  Context ctx = create_largek_context();

  ctx.initial_partitioning.pool.min_num_repetitions = 2;
  ctx.initial_partitioning.pool.min_num_non_adaptive_repetitions = 1;
  ctx.initial_partitioning.pool.max_num_repetitions = 2;
  ctx.initial_partitioning.pool.enable_bfs_bipartitioner = true;
  ctx.initial_partitioning.pool.enable_ggg_bipartitioner = false;
  ctx.initial_partitioning.pool.enable_random_bipartitioner = true;

  ctx.initial_partitioning.pool.refinement.algorithms = {InitialRefinementAlgorithm::NOOP};
  ctx.initial_partitioning.pool.refinement.fm.num_iterations = 1;

  ctx.initial_partitioning.refine_pool_partition = true;

  return ctx;
}

Context create_largek_strong_context() {
  Context ctx = create_largek_context();

  ctx.refinement.algorithms = {
      RefinementAlgorithm::GREEDY_BALANCER,
      RefinementAlgorithm::LABEL_PROPAGATION,
      RefinementAlgorithm::KWAY_FM,
      RefinementAlgorithm::GREEDY_BALANCER,
  };

  ctx.refinement.kway_fm.gain_cache_strategy = GainCacheStrategy::COMPACT_HASHING_LARGE_K;

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
  ctx.refinement.algorithms = {};
  return ctx;
}

namespace {

inline Context terapartify_context(Context ctx) {
  ctx.compression.enabled = true;
  ctx.partitioning.deep_initial_partitioning_mode = DeepInitialPartitioningMode::SEQUENTIAL;
  return ctx;
}

} // namespace

Context create_terapart_context() {
  return terapartify_context(create_default_context());
}

Context create_terapart_strong_context() {
  return terapartify_context(create_strong_context());
}

Context create_terapart_largek_context() {
  Context ctx = terapartify_context(create_largek_context());
  ctx.coarsening.clustering.forced_kc_level = true;
  return ctx;
}

Context create_vcycle_context(const bool restrict_refinement) {
  Context ctx = create_default_context();
  ctx.partitioning.mode = PartitioningMode::VCYCLE;

  if (restrict_refinement) {
    ctx.partitioning.restrict_vcycle_refinement = true;
    ctx.refinement.algorithms = {
        // GREEDY_BALANCER does not respect the community structure
        RefinementAlgorithm::LABEL_PROPAGATION,
    };
  }

  return ctx;
}

Context create_esa21_smallk_context() {
  Context ctx = create_default_context();

  ctx.coarsening.contraction.algorithm = ContractionAlgorithm::BUFFERED;
  ctx.coarsening.clustering.lp.impl = LabelPropagationImplementation::SINGLE_PHASE;

  return ctx;
}

Context create_esa21_largek_context() {
  Context ctx = create_esa21_smallk_context();

  ctx.initial_partitioning.pool.min_num_repetitions = 4;
  ctx.initial_partitioning.pool.min_num_non_adaptive_repetitions = 2;
  ctx.initial_partitioning.pool.max_num_repetitions = 4;

  return ctx;
}

Context create_esa21_largek_fast_context() {
  Context ctx = create_esa21_largek_context();

  ctx.initial_partitioning.pool.min_num_repetitions = 2;
  ctx.initial_partitioning.pool.min_num_non_adaptive_repetitions = 1;
  ctx.initial_partitioning.pool.max_num_repetitions = 2;
  ctx.initial_partitioning.pool.enable_bfs_bipartitioner = true;
  ctx.initial_partitioning.pool.enable_ggg_bipartitioner = false;
  ctx.initial_partitioning.pool.enable_random_bipartitioner = true;

  ctx.initial_partitioning.pool.refinement.algorithms = {InitialRefinementAlgorithm::NOOP};
  ctx.initial_partitioning.pool.refinement.fm.num_iterations = 1;

  ctx.initial_partitioning.refine_pool_partition = true;

  return ctx;
}

Context create_esa21_strong_context() {
  Context ctx = create_esa21_smallk_context();

  ctx.refinement.algorithms = {
      RefinementAlgorithm::GREEDY_BALANCER,
      RefinementAlgorithm::LABEL_PROPAGATION,
      RefinementAlgorithm::KWAY_FM,
      RefinementAlgorithm::GREEDY_BALANCER,
  };

  return ctx;
}

// Configures the coarsening phase similar to Mt-KaHyPar, plus configures direct k-way initial
// partitioning instead of deep MGP.
Context create_mtkahypar_kway_context() {
  Context ctx = create_default_context();
  ctx.coarsening.clustering.lp.num_iterations = 1;
  ctx.coarsening.clustering.cluster_weight_limit = ClusterWeightLimit::BLOCK_WEIGHT;
  ctx.coarsening.clustering.cluster_weight_multiplier = 1.0 / 160.0;
  ctx.coarsening.clustering.shrink_factor = 2.5;
  ctx.coarsening.contraction_limit = 160;
  ctx.coarsening.clustering.lp.two_hop_strategy = TwoHopStrategy::CLUSTER;
  ctx.partitioning.mode = PartitioningMode::KWAY;
  return ctx;
}

// Based on Mt-KaHyPar coarsening, uses edge sparsification for worst-case linear-time running time.
Context create_linear_time_kway_context() {
  Context ctx = create_mtkahypar_kway_context();
  ctx.coarsening.algorithm = CoarseningAlgorithm::SPARSIFICATION_CLUSTERING;
  return ctx;
}

} // namespace kaminpar::shm
