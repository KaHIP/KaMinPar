/*******************************************************************************
 * Utility functions to read/write parts of the partitioner context from/to
 * strings.
 *
 * @file:   context_io.cc
 * @author: Daniel Seemaier
 * @date:   27.10.2022
 ******************************************************************************/
#include "kaminpar-dist/context_io.h"

#include <iomanip>
#include <ostream>
#include <unordered_map>

#include "kaminpar-dist/context.h"

#include "kaminpar-common/console_io.h"
#include "kaminpar-common/random.h"

namespace kaminpar::dist {
namespace {
template <typename T> std::ostream &operator<<(std::ostream &out, const std::vector<T> &vec) {
  out << "[";
  bool first = true;
  for (const T &e : vec) {
    if (first) {
      first = false;
    } else {
      out << " -> ";
    }
    out << e;
  }
  return out << "]";
}
} // namespace

std::unordered_map<std::string, PartitioningMode> get_partitioning_modes() {
  return {
      {"multilevel/deep", PartitioningMode::DEEP},
      {"multilevel/kway", PartitioningMode::KWAY},
  };
}

std::ostream &operator<<(std::ostream &out, const PartitioningMode mode) {
  switch (mode) {
  case PartitioningMode::DEEP:
    return out << "multilevel/deep";
  case PartitioningMode::KWAY:
    return out << "multilevel/kway";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, GlobalClusteringAlgorithm> get_global_clustering_algorithms() {
  return {
      {"noop", GlobalClusteringAlgorithm::NOOP},
      {"lp", GlobalClusteringAlgorithm::LP},
      {"hem", GlobalClusteringAlgorithm::HEM},
      {"hem-lp", GlobalClusteringAlgorithm::HEM_LP},
  };
}

std::ostream &operator<<(std::ostream &out, const GlobalClusteringAlgorithm algorithm) {
  switch (algorithm) {
  case GlobalClusteringAlgorithm::NOOP:
    return out << "noop";
  case GlobalClusteringAlgorithm::LP:
    return out << "lp";
  case GlobalClusteringAlgorithm::HEM:
    return out << "hem";
  case GlobalClusteringAlgorithm::HEM_LP:
    return out << "hem-lp";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, LocalClusteringAlgorithm> get_local_clustering_algorithms() {
  return {
      {"noop", LocalClusteringAlgorithm::NOOP},
      {"lp", LocalClusteringAlgorithm::LP},
  };
}

std::ostream &operator<<(std::ostream &out, const LocalClusteringAlgorithm algorithm) {
  switch (algorithm) {
  case LocalClusteringAlgorithm::NOOP:
    return out << "noop";
  case LocalClusteringAlgorithm::LP:
    return out << "lp";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, InitialPartitioningAlgorithm>
get_initial_partitioning_algorithms() {
  return {
      {"kaminpar", InitialPartitioningAlgorithm::KAMINPAR},
      {"mtkahypar", InitialPartitioningAlgorithm::MTKAHYPAR},
      {"random", InitialPartitioningAlgorithm::RANDOM},
  };
}

std::ostream &operator<<(std::ostream &out, const InitialPartitioningAlgorithm algorithm) {
  switch (algorithm) {
  case InitialPartitioningAlgorithm::KAMINPAR:
    return out << "kaminpar";
  case InitialPartitioningAlgorithm::MTKAHYPAR:
    return out << "mtkahypar";
  case InitialPartitioningAlgorithm::RANDOM:
    return out << "random";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, RefinementAlgorithm> get_kway_refinement_algorithms() {
  return {
      {"noop", RefinementAlgorithm::NOOP},

      {"lp", RefinementAlgorithm::BATCHED_LP_REFINER},
      {"blp", RefinementAlgorithm::BATCHED_LP_REFINER},
      {"batched-lp", RefinementAlgorithm::BATCHED_LP_REFINER},

      {"clp", RefinementAlgorithm::COLORED_LP_REFINER},
      {"colored-lp", RefinementAlgorithm::COLORED_LP_REFINER},

      {"nb", RefinementAlgorithm::HYBRID_NODE_BALANCER},
      {"hybrid-node-balancer", RefinementAlgorithm::HYBRID_NODE_BALANCER},

      {"cb", RefinementAlgorithm::HYBRID_CLUSTER_BALANCER},
      {"hybrid-cluster-balancer", RefinementAlgorithm::HYBRID_CLUSTER_BALANCER},

      {"jet", RefinementAlgorithm::JET_REFINER},
      {"mtkahypar", RefinementAlgorithm::MTKAHYPAR_REFINER},
  };
}

std::unordered_map<std::string, RefinementAlgorithm> get_balancing_algorithms() {
  return {
      {"noop", RefinementAlgorithm::NOOP},
      {"hybrid-node-balancer", RefinementAlgorithm::HYBRID_NODE_BALANCER},
      {"hybrid-cluster-balancer", RefinementAlgorithm::HYBRID_CLUSTER_BALANCER},
      {"mtkahypar", RefinementAlgorithm::MTKAHYPAR_REFINER},
  };
};

std::ostream &operator<<(std::ostream &out, const RefinementAlgorithm algorithm) {
  switch (algorithm) {
  case RefinementAlgorithm::NOOP:
    return out << "noop";
  case RefinementAlgorithm::BATCHED_LP_REFINER:
    return out << "batched-lp";
  case RefinementAlgorithm::COLORED_LP_REFINER:
    return out << "colored-lp";
  case RefinementAlgorithm::JET_REFINER:
    return out << "jet";
  case RefinementAlgorithm::MTKAHYPAR_REFINER:
    return out << "mtkahypar";
  case RefinementAlgorithm::HYBRID_NODE_BALANCER:
    return out << "hybrid-node-balancer";
  case RefinementAlgorithm::HYBRID_CLUSTER_BALANCER:
    return out << "hybrid-cluster-balancer";
  }

  return out << "<invalid>";
}

std::string get_refinement_algorithms_description() {
  return std::string(R"(
- noop:                    disable refinement
- batched-lp:              Label propagation with synchronization after a constant number of supersteps
- colored-lp:              Label propagation with synchronization after each color class
- jet:                     Jet refinement)")
             .substr(1) +
         "\n" + get_balancing_algorithms_description();
}

std::string get_balancing_algorithms_description() {
  return std::string(R"(
- hybrid-node-balancer:    hybrid node balancer)")
      .substr(1);
}

std::unordered_map<std::string, LabelPropagationMoveExecutionStrategy>
get_label_propagation_move_execution_strategies() {
  return {
      {"probabilistic", LabelPropagationMoveExecutionStrategy::PROBABILISTIC},
      {"best", LabelPropagationMoveExecutionStrategy::BEST_MOVES},
      {"local", LabelPropagationMoveExecutionStrategy::LOCAL_MOVES},
  };
}

std::ostream &operator<<(std::ostream &out, const LabelPropagationMoveExecutionStrategy strategy) {
  switch (strategy) {
  case LabelPropagationMoveExecutionStrategy::PROBABILISTIC:
    return out << "probabilistic";
  case LabelPropagationMoveExecutionStrategy::BEST_MOVES:
    return out << "best";
  case LabelPropagationMoveExecutionStrategy::LOCAL_MOVES:
    return out << "local";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, GraphOrdering> get_graph_orderings() {
  return {
      {"natural", GraphOrdering::NATURAL},
      {"deg-buckets", GraphOrdering::DEGREE_BUCKETS},
      {"degree-buckets", GraphOrdering::DEGREE_BUCKETS},
      {"coloring", GraphOrdering::COLORING},
  };
}

std::ostream &operator<<(std::ostream &out, const GraphOrdering ordering) {
  switch (ordering) {
  case GraphOrdering::NATURAL:
    return out << "natural";
  case GraphOrdering::DEGREE_BUCKETS:
    return out << "deg-buckets";
  case GraphOrdering::COLORING:
    return out << "coloring";
  }

  return out << "<invalid>";
}

std::ostream &operator<<(std::ostream &out, const ClusterSizeStrategy strategy) {
  switch (strategy) {
  case ClusterSizeStrategy::ZERO:
    return out << "zero";
  case ClusterSizeStrategy::ONE:
    return out << "one";
  case ClusterSizeStrategy::MAX_OVERLOAD:
    return out << "max-overload";
  case ClusterSizeStrategy::AVG_OVERLOAD:
    return out << "avg-overload";
  case ClusterSizeStrategy::MIN_OVERLOAD:
    return out << "min-overload";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, ClusterSizeStrategy> get_move_set_size_strategies() {
  return {
      {"zero", ClusterSizeStrategy::ZERO},
      {"one", ClusterSizeStrategy::ONE},
      {"max-overload", ClusterSizeStrategy::MAX_OVERLOAD},
      {"avg-overload", ClusterSizeStrategy::AVG_OVERLOAD},
      {"min-overload", ClusterSizeStrategy::MIN_OVERLOAD},
  };
}

std::ostream &operator<<(std::ostream &out, const ClusterStrategy strategy) {
  switch (strategy) {
  case ClusterStrategy::SINGLETONS:
    return out << "singletons";
  case ClusterStrategy::LP:
    return out << "lp";
  case ClusterStrategy::GREEDY_BATCH_PREFIX:
    return out << "greedy-batch-prefix";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, ClusterStrategy> get_move_set_strategies() {
  return {
      {"singletons", ClusterStrategy::SINGLETONS},
      {"lp", ClusterStrategy::LP},
      {"greedy-batch-prefix", ClusterStrategy::GREEDY_BATCH_PREFIX},
  };
}

void print(const Context &ctx, const bool root, std::ostream &out, MPI_Comm comm) {
  if (root) {
    out << "Seed:                         " << Random::get_seed() << "\n";
    out << "Graph:                        " << ctx.debug.graph_filename << "\n";
    out << "  Rearrange graph by:         " << ctx.rearrange_by << "\n";
  }
  print(ctx.partition, root, out, comm);
  if (root) {
    cio::print_delimiter("Partitioning Scheme", '-');

    out << "Partitioning mode:            " << ctx.mode << "\n";
    if (ctx.mode == PartitioningMode::DEEP) {
      out << "  Enable PE-splitting:        " << (ctx.enable_pe_splitting ? "yes" : "no") << "\n";
      out << "  Partition extension factor: " << ctx.partition.K << "\n";
      out << "  Simulate seq. hybrid exe.:  " << (ctx.simulate_singlethread ? "yes" : "no") << "\n";
    }
    cio::print_delimiter("Coarsening", '-');
    print(ctx.coarsening, ctx.parallel, out);
    cio::print_delimiter("Initial Partitioning", '-');
    print(ctx.initial_partitioning, out);
    cio::print_delimiter("Refinement", '-');
    print(ctx.refinement, ctx.parallel, out);
  }
}

void print(const PartitionContext &ctx, const bool root, std::ostream &out, MPI_Comm comm) {
  // If the graph context has not been initialized with a graph, be silent
  // (This should never happen)
  if (ctx.graph == nullptr) {
    return;
  }

  const auto size = std::max<std::uint64_t>({
      static_cast<std::uint64_t>(ctx.graph->global_n),
      static_cast<std::uint64_t>(ctx.graph->global_m),
      static_cast<std::uint64_t>(ctx.graph->max_block_weight(0)),
  });
  const auto width = std::ceil(std::log10(size)) + 1;

  const auto num_global_total_nodes =
      mpi::allreduce<GlobalNodeID>(ctx.graph->total_n, MPI_SUM, comm);

  if (root) {
    out << "  Number of global nodes:    " << std::setw(width) << ctx.graph->global_n;
    if (asserting_cast<GlobalNodeWeight>(ctx.graph->global_n) ==
        ctx.graph->global_total_node_weight) {
      out << " (unweighted)\n";
    } else {
      out << " (total weight: " << ctx.graph->global_total_node_weight << ")\n";
    }
    out << "    + ghost nodes:           " << std::setw(width)
        << num_global_total_nodes - ctx.graph->global_n << "\n";
    out << "  Number of global edges:    " << std::setw(width) << ctx.graph->global_m;
    if (asserting_cast<GlobalEdgeWeight>(ctx.graph->global_m) ==
        ctx.graph->global_total_edge_weight) {
      out << " (unweighted)\n";
    } else {
      out << " (total weight: " << ctx.graph->global_total_edge_weight << ")\n";
    }
    out << "Number of blocks:             " << ctx.k << "\n";
    out << "Maximum block weight:         " << ctx.graph->max_block_weight(0) << " ("
        << ctx.graph->perfectly_balanced_block_weight(0) << " + " << 100 * ctx.epsilon << "%)\n";
  }
}

void print(const ChunksContext &ctx, const ParallelContext &parallel, std::ostream &out) {
  if (ctx.fixed_num_chunks == 0) {
    out << "  Number of chunks:           " << ctx.compute(parallel) << "[= max("
        << ctx.min_num_chunks << ", " << ctx.total_num_chunks << " / " << parallel.num_mpis
        << (ctx.scale_chunks_with_threads
                ? std::string(" / ") + std::to_string(parallel.num_threads)
                : "")
        << "]\n";
  } else {
    out << "  Number of chunks:           " << ctx.fixed_num_chunks << "\n";
  }
}

void print(const CoarseningContext &ctx, const ParallelContext &parallel, std::ostream &out) {
  out << "Contraction limit:            " << ctx.contraction_limit << "\n";
  if (ctx.max_global_clustering_levels > 0 && ctx.max_local_clustering_levels > 0) {
    out << "Coarsening mode:              local[" << ctx.max_local_clustering_levels << "]+global["
        << ctx.max_global_clustering_levels << "]\n";
  } else if (ctx.max_global_clustering_levels > 0) {
    out << "Coarsening mode:              global[" << ctx.max_global_clustering_levels << "]\n";
  } else if (ctx.max_local_clustering_levels > 0) {
    out << "Coarsening mode:              local[" << ctx.max_local_clustering_levels << "]\n";
  } else {
    out << "Coarsening mode:              disabled\n";
  }

  if (ctx.max_local_clustering_levels > 0) {
    out << "Local clustering algorithm:   " << ctx.local_clustering_algorithm << "\n";
    out << "  Number of iterations:       " << ctx.local_lp.num_iterations << "\n";
    out << "  High degree threshold:      " << ctx.local_lp.passive_high_degree_threshold
        << " (passive), " << ctx.local_lp.active_high_degree_threshold << " (active)\n";
    out << "  Max degree:                 " << ctx.local_lp.max_num_neighbors << "\n";
    out << "  Ghost nodes:                "
        << (ctx.local_lp.ignore_ghost_nodes ? "ignore" : "consider") << "+"
        << (ctx.local_lp.keep_ghost_clusters ? "keep" : "discard") << "\n";
  }

  if (ctx.max_global_clustering_levels > 0) {
    out << "Global clustering algorithm:  " << ctx.global_clustering_algorithm;
    if (ctx.max_cnode_imbalance < std::numeric_limits<double>::max()) {
      out << " [rebalance if >" << std::setprecision(2) << 100.0 * (ctx.max_cnode_imbalance - 1.0)
          << "%";
      if (ctx.migrate_cnode_prefix) {
        out << ", prefix";
      } else {
        out << ", suffix";
      }
      if (ctx.force_perfect_cnode_balance) {
        out << ", strict";
      } else {
        out << ", relaxed";
      }
      out << "]";
    } else {
      out << "[natural assignment]";
    }
    out << "\n";

    if (ctx.global_clustering_algorithm == GlobalClusteringAlgorithm::LP ||
        ctx.global_clustering_algorithm == GlobalClusteringAlgorithm::HEM_LP) {
      out << "  Number of iterations:       " << ctx.global_lp.num_iterations << "\n";
      out << "  High degree threshold:      " << ctx.global_lp.passive_high_degree_threshold
          << " (passive), " << ctx.global_lp.active_high_degree_threshold << " (active)\n";
      out << "  Max degree:                 " << ctx.global_lp.max_num_neighbors << "\n";
      print(ctx.global_lp.chunks, parallel, out);
      out << "  Active set:                 "
          << (ctx.global_clustering_algorithm == GlobalClusteringAlgorithm::LP ? "no" : "yes")
          << "\n";
      out << "  Cluster weights:            "
          << (ctx.global_lp.sync_cluster_weights ? "sync" : "no-sync") << "+"
          << (ctx.global_lp.enforce_cluster_weights ? "enforce" : "no-enforce") << " "
          << (ctx.global_lp.cheap_toplevel ? "(on level > 1)" : "(always)") << "\n";
    }

    if (ctx.global_clustering_algorithm == GlobalClusteringAlgorithm::HEM ||
        ctx.global_clustering_algorithm == GlobalClusteringAlgorithm::HEM_LP) {
      print(ctx.hem.chunks, parallel, out);
      out << "  Small color blacklist:      " << 100 * ctx.hem.small_color_blacklist << "%"
          << (ctx.hem.only_blacklist_input_level ? " (input level only)" : "") << "\n";
    }
  }
}

void print(const InitialPartitioningContext &ctx, std::ostream &out) {
  out << "IP algorithm:                 " << ctx.algorithm << "\n";
  if (ctx.algorithm == InitialPartitioningAlgorithm::KAMINPAR) {
    out << "  Configuration preset:       default\n";
  }
}

void print(const RefinementContext &ctx, const ParallelContext &parallel, std::ostream &out) {
  out << "Refinement algorithms:        " << ctx.algorithms << "\n";
  out << "Refine initial partition:     " << (ctx.refine_coarsest_level ? "yes" : "no") << "\n";
  if (ctx.includes_algorithm(RefinementAlgorithm::BATCHED_LP_REFINER)) {
    out << "Label propagation:            " << RefinementAlgorithm::BATCHED_LP_REFINER << "\n";
    out << "  Number of iterations:       " << ctx.lp.num_iterations << "\n";
    print(ctx.lp.chunks, parallel, out);
    out << "  Use probabilistic moves:    " << (ctx.lp.ignore_probabilities ? "no" : "yes") << "\n";
    out << "  Number of retries:          " << ctx.lp.num_move_attempts << "\n";
  }
  if (ctx.includes_algorithm(RefinementAlgorithm::COLORED_LP_REFINER)) {
    out << "Colored Label Propagation:    " << RefinementAlgorithm::COLORED_LP_REFINER << "\n";
    out << "  Number of iterations:       " << ctx.colored_lp.num_iterations << "\n";
    print(ctx.colored_lp.coloring_chunks, parallel, out);
    out << "  Commitment strategy:        " << ctx.colored_lp.move_execution_strategy << "\n";
    if (ctx.colored_lp.move_execution_strategy ==
        LabelPropagationMoveExecutionStrategy::PROBABILISTIC) {
      out << "    Number of attempts:       " << ctx.colored_lp.num_probabilistic_move_attempts
          << "\n";
    } else if (ctx.colored_lp.move_execution_strategy == LabelPropagationMoveExecutionStrategy::BEST_MOVES) {
      out << "    Sort by:                  "
          << (ctx.colored_lp.sort_by_rel_gain ? "relative gain" : "absolute gain") << "\n";
    }
    out << "  Commitment rounds:          " << ctx.colored_lp.num_move_execution_iterations << "\n";
    out << "  Track block weights:        "
        << (ctx.colored_lp.track_local_block_weights ? "yes" : "no") << "\n";
    out << "  Use active set:             " << (ctx.colored_lp.use_active_set ? "yes" : "no")
        << "\n";
    out << "  Small color blacklist:      " << 100 * ctx.colored_lp.small_color_blacklist << "%"
        << (ctx.colored_lp.only_blacklist_input_level ? " (input level only)" : "") << "\n";
  }
  if (ctx.includes_algorithm(RefinementAlgorithm::JET_REFINER)) {
    out << "Jet refinement:               " << RefinementAlgorithm::JET_REFINER << "\n";
    out << "  Number of rounds:           coarse " << ctx.jet.num_coarse_rounds << ", fine "
        << ctx.jet.num_fine_rounds << "\n";
    out << "  Number of iterations:       max " << ctx.jet.num_iterations << ", or "
        << ctx.jet.num_fruitless_iterations << " fruitless (improvement < "
        << 100.0 * (1 - ctx.jet.fruitless_threshold) << "%)\n";
    out << "  Negative gain factors:      "
        << (ctx.jet.dynamic_negative_gain_factor ? "dynamic" : "static") << "\n";
    out << "  Static factors:             coarse " << ctx.jet.coarse_negative_gain_factor
        << ", fine " << ctx.jet.fine_negative_gain_factor << "\n";
    out << "  Dynamic factors:            initial " << ctx.jet.initial_negative_gain_factor
        << ", final " << ctx.jet.final_negative_gain_factor << "\n";
    out << "  Balancing algorithm:        " << ctx.jet.balancing_algorithm << "\n";
  }
  if (ctx.includes_algorithm(RefinementAlgorithm::HYBRID_NODE_BALANCER) ||
      (ctx.includes_algorithm(RefinementAlgorithm::JET_REFINER) &&
       ctx.jet.balancing_algorithm == RefinementAlgorithm::HYBRID_NODE_BALANCER)) {
    out << "Node balancer:                " << RefinementAlgorithm::HYBRID_NODE_BALANCER << "\n";
    out << "  Number of rounds:           " << ctx.node_balancer.max_num_rounds << "\n";
    out << "  Sequential balancing:       "
        << (ctx.node_balancer.enable_sequential_balancing ? "yes" : "no") << "\n";
    out << "    Nodes per block:          " << ctx.node_balancer.seq_num_nodes_per_block << "\n";
    out << "  Parallel balancing:         "
        << (ctx.node_balancer.enable_parallel_balancing ? "yes" : "no") << "\n";
    out << "    Threshold:                " << ctx.node_balancer.par_threshold << "\n";
    out << "    # of dicing attempts:     " << ctx.node_balancer.par_num_dicing_attempts << " --> "
        << (ctx.node_balancer.par_accept_imbalanced_moves ? "accept" : "reject") << "\n";
    out << "    Gain buckets:             base " << ctx.node_balancer.par_gain_bucket_base
        << ", positive gain buckets: "
        << (ctx.node_balancer.par_enable_positive_gain_buckets ? "yes" : "no") << "\n";
    out << "    Partial buckets:          "
        << (ctx.node_balancer.par_partial_buckets ? "yes" : "no") << "\n";
    out << "    Update PQ during build:   "
        << (ctx.node_balancer.par_update_pq_gains ? "yes" : "no") << "\n";
    out << "    High degree thresholds:   insertions = "
        << ctx.node_balancer.par_high_degree_insertion_threshold
        << ", updates = " << ctx.node_balancer.par_high_degree_update_thresold
        << " [interval: " << ctx.node_balancer.par_high_degree_update_interval << "]\n";
  }
  if (ctx.includes_algorithm(RefinementAlgorithm::HYBRID_CLUSTER_BALANCER) ||
      (ctx.includes_algorithm(RefinementAlgorithm::JET_REFINER) &&
       ctx.jet.balancing_algorithm == RefinementAlgorithm::HYBRID_CLUSTER_BALANCER)) {
    out << "Cluster balancer:             " << RefinementAlgorithm::HYBRID_CLUSTER_BALANCER << "\n";
    out << "  Clusters:                   " << ctx.cluster_balancer.cluster_strategy << "\n";
    out << "    Max weight:               " << ctx.cluster_balancer.cluster_size_strategy << " x "
        << ctx.cluster_balancer.cluster_size_multiplier << "\n";
    out << "    Rebuild interval:         "
        << (ctx.cluster_balancer.cluster_rebuild_interval == 0
                ? "never"
                : std::string("every ") +
                      std::to_string(ctx.cluster_balancer.cluster_rebuild_interval) + " round(s)")
        << "\n";
    out << "  Maximum number of rounds:   " << ctx.cluster_balancer.max_num_rounds << "\n";
    out << "  Sequential balancing:       "
        << (ctx.cluster_balancer.enable_sequential_balancing ? "enabled" : "disabled") << "\n";
    out << "    No. of nodes per block:   " << ctx.cluster_balancer.seq_num_nodes_per_block << "\n";
    out << "    Keep all nodes in PQ:     " << (ctx.cluster_balancer.seq_full_pq ? "yes" : "no")
        << "\n";
    out << "  Parallel balancing:         "
        << (ctx.cluster_balancer.enable_parallel_balancing ? "enabled" : "disabled") << "\n";
    out << "    Trigger threshold:        " << ctx.cluster_balancer.parallel_threshold << "\n";
    out << "    # of dicing attempts:     " << ctx.cluster_balancer.par_num_dicing_attempts
        << " --> " << (ctx.cluster_balancer.par_accept_imbalanced ? "accept" : "reject") << "\n";
    out << "    Gain buckets:             log" << ctx.cluster_balancer.par_gain_bucket_factor
        << ", positive gain buckets: "
        << (ctx.cluster_balancer.par_use_positive_gain_buckets ? "yes" : "no") << "\n";
    out << "    Parallel rebalancing:     start at "
        << 100.0 * ctx.cluster_balancer.par_initial_rebalance_fraction << "%, increase by "
        << 100.0 * ctx.cluster_balancer.par_rebalance_fraction_increase << "% each round\n";
  }
}
} // namespace kaminpar::dist
