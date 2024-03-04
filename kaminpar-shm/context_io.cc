/*******************************************************************************
 * IO functions for the context structs.
 *
 * @file:   context_io.cc
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 ******************************************************************************/
#include "kaminpar-shm/context_io.h"

#include <algorithm>
#include <cmath>
#include <iomanip>

#include "kaminpar-common/asserting_cast.h"
#include "kaminpar-common/console_io.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/strutils.h"

namespace kaminpar::shm {
using namespace std::string_literals;

std::unordered_map<std::string, GraphOrdering> get_graph_orderings() {
  return {
      {"natural", GraphOrdering::NATURAL},
      {"deg-buckets", GraphOrdering::DEGREE_BUCKETS},
      {"degree-buckets", GraphOrdering::DEGREE_BUCKETS},
  };
}

std::ostream &operator<<(std::ostream &out, const GraphOrdering ordering) {
  switch (ordering) {
  case GraphOrdering::NATURAL:
    return out << "natural";
  case GraphOrdering::DEGREE_BUCKETS:
    return out << "deg-buckets";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, ClusteringAlgorithm> get_clustering_algorithms() {
  return {
      {"noop", ClusteringAlgorithm::NOOP},
      {"lp", ClusteringAlgorithm::LABEL_PROPAGATION},
  };
}

std::ostream &operator<<(std::ostream &out, const ClusteringAlgorithm algorithm) {
  switch (algorithm) {
  case ClusteringAlgorithm::NOOP:
    return out << "noop";
  case ClusteringAlgorithm::LABEL_PROPAGATION:
    return out << "lp";
  }
  return out << "<invalid>";
}

std::unordered_map<std::string, ClusterWeightLimit> get_cluster_weight_limits() {
  return {
      {"epsilon-block-weight", ClusterWeightLimit::EPSILON_BLOCK_WEIGHT},
      {"static-block-weight", ClusterWeightLimit::BLOCK_WEIGHT},
      {"one", ClusterWeightLimit::ONE},
      {"zero", ClusterWeightLimit::ZERO},
  };
}

std::ostream &operator<<(std::ostream &out, const ClusterWeightLimit limit) {
  switch (limit) {
  case ClusterWeightLimit::EPSILON_BLOCK_WEIGHT:
    return out << "epsilon-block-weight";
  case ClusterWeightLimit::BLOCK_WEIGHT:
    return out << "static-block-weight";
  case ClusterWeightLimit::ONE:
    return out << "one";
  case ClusterWeightLimit::ZERO:
    return out << "zero";
  }
  return out << "<invalid>";
}

std::unordered_map<std::string, RefinementAlgorithm> get_kway_refinement_algorithms() {
  return {
      {"noop", RefinementAlgorithm::NOOP},
      {"lp", RefinementAlgorithm::LABEL_PROPAGATION},
      {"fm", RefinementAlgorithm::KWAY_FM},
      {"jet", RefinementAlgorithm::JET},
      {"greedy-balancer", RefinementAlgorithm::GREEDY_BALANCER},
      {"mtkahypar", RefinementAlgorithm::MTKAHYPAR},
  };
}

std::ostream &operator<<(std::ostream &out, const RefinementAlgorithm algorithm) {
  switch (algorithm) {
  case RefinementAlgorithm::NOOP:
    return out << "noop";
  case RefinementAlgorithm::KWAY_FM:
    return out << "fm";
  case RefinementAlgorithm::LABEL_PROPAGATION:
    return out << "lp";
  case RefinementAlgorithm::GREEDY_BALANCER:
    return out << "greedy-balancer";
  case RefinementAlgorithm::JET:
    return out << "jet";
  case RefinementAlgorithm::MTKAHYPAR:
    return out << "mtkahypar";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, FMStoppingRule> get_fm_stopping_rules() {
  return {
      {"simple", FMStoppingRule::SIMPLE},
      {"adaptive", FMStoppingRule::ADAPTIVE},
  };
}

std::ostream &operator<<(std::ostream &out, const FMStoppingRule rule) {
  switch (rule) {
  case FMStoppingRule::SIMPLE:
    return out << "simple";
  case FMStoppingRule::ADAPTIVE:
    return out << "adaptive";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, PartitioningMode> get_partitioning_modes() {
  return {
      {"deep", PartitioningMode::DEEP},
      {"rb", PartitioningMode::RB},
      {"kway", PartitioningMode::KWAY},
  };
}

std::ostream &operator<<(std::ostream &out, const PartitioningMode mode) {
  switch (mode) {
  case PartitioningMode::DEEP:
    return out << "deep";
  case PartitioningMode::RB:
    return out << "rb";
  case PartitioningMode::KWAY:
    return out << "kway";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, InitialPartitioningMode> get_initial_partitioning_modes() {
  return {
      {"sequential", InitialPartitioningMode::SEQUENTIAL},
      {"async-parallel", InitialPartitioningMode::ASYNCHRONOUS_PARALLEL},
      {"sync-parallel", InitialPartitioningMode::SYNCHRONOUS_PARALLEL},
  };
}

std::ostream &operator<<(std::ostream &out, const InitialPartitioningMode mode) {
  switch (mode) {
  case InitialPartitioningMode::SEQUENTIAL:
    return out << "sequential";
  case InitialPartitioningMode::ASYNCHRONOUS_PARALLEL:
    return out << "async-parallel";
  case InitialPartitioningMode::SYNCHRONOUS_PARALLEL:
    return out << "sync-parallel";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, GainCacheStrategy> get_gain_cache_strategies() {
  return {
      {"sparse", GainCacheStrategy::SPARSE},
      {"dense", GainCacheStrategy::DENSE},
      {"on-the-fly", GainCacheStrategy::ON_THE_FLY},
      {"hybrid", GainCacheStrategy::HYBRID},
      {"tracing", GainCacheStrategy::TRACING},
  };
}

std::ostream &operator<<(std::ostream &out, const GainCacheStrategy strategy) {
  switch (strategy) {
  case GainCacheStrategy::SPARSE:
    return out << "sparse";
  case GainCacheStrategy::DENSE:
    return out << "dense";
  case GainCacheStrategy::ON_THE_FLY:
    return out << "on-the-fly";
  case GainCacheStrategy::HYBRID:
    return out << "hybrid";
  case GainCacheStrategy::TRACING:
    return out << "tracing";
  }

  return out << "<invalid>";
}

std::ostream &operator<<(std::ostream &out, const TwoHopStrategy strategy) {
  switch (strategy) {
  case TwoHopStrategy::DISABLE:
    return out << "disable";
  case TwoHopStrategy::MATCH:
    return out << "match";
  case TwoHopStrategy::MATCH_THREADWISE:
    return out << "match-threadwise";
  case TwoHopStrategy::CLUSTER:
    return out << "cluster";
  case TwoHopStrategy::CLUSTER_THREADWISE:
    return out << "cluster-threadwise";
  case TwoHopStrategy::LEGACY:
    return out << "legacy";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, TwoHopStrategy> get_two_hop_strategies() {
  return {
      {"disable", TwoHopStrategy::DISABLE},
      {"match", TwoHopStrategy::MATCH},
      {"match-threadwise", TwoHopStrategy::MATCH_THREADWISE},
      {"cluster", TwoHopStrategy::CLUSTER},
      {"cluster-threadwise", TwoHopStrategy::CLUSTER_THREADWISE},
      {"legacy", TwoHopStrategy::LEGACY},
  };
}

std::ostream &operator<<(std::ostream &out, IsolatedNodesClusteringStrategy strategy) {
  switch (strategy) {
  case IsolatedNodesClusteringStrategy::KEEP:
    return out << "keep";
  case IsolatedNodesClusteringStrategy::MATCH:
    return out << "match";
  case IsolatedNodesClusteringStrategy::CLUSTER:
    return out << "cluster";
  case IsolatedNodesClusteringStrategy::MATCH_DURING_TWO_HOP:
    return out << "match-during-two-hop";
  case IsolatedNodesClusteringStrategy::CLUSTER_DURING_TWO_HOP:
    return out << "cluster-during-two-hop";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, IsolatedNodesClusteringStrategy>
get_isolated_nodes_clustering_strategies() {
  return {
      {"keep", IsolatedNodesClusteringStrategy::KEEP},
      {"match", IsolatedNodesClusteringStrategy::MATCH},
      {"cluster", IsolatedNodesClusteringStrategy::CLUSTER},
      {"match-during-two-hop", IsolatedNodesClusteringStrategy::MATCH_DURING_TWO_HOP},
      {"cluster-during-two-hop", IsolatedNodesClusteringStrategy::CLUSTER_DURING_TWO_HOP},
  };
}

void print(const CoarseningContext &c_ctx, std::ostream &out) {
  out << "Contraction limit:            " << c_ctx.contraction_limit << "\n";
  out << "Cluster weight limit:         " << c_ctx.cluster_weight_limit << " x "
      << c_ctx.cluster_weight_multiplier << "\n";
  out << "Clustering algorithm:         " << c_ctx.algorithm << "\n";
  if (c_ctx.algorithm == ClusteringAlgorithm::LABEL_PROPAGATION) {
    print(c_ctx.lp, out);
  }
}

void print(const LabelPropagationCoarseningContext &lp_ctx, std::ostream &out) {
  out << "  Number of iterations:       " << lp_ctx.num_iterations << "\n";
  out << "  High degree threshold:      " << lp_ctx.large_degree_threshold << "\n";
  out << "  Max degree:                 " << lp_ctx.max_num_neighbors << "\n";
  out << "  2-hop clustering:           " << lp_ctx.two_hop_strategy << ", if |Vcoarse| > "
      << std::setw(2) << std::fixed << lp_ctx.two_hop_threshold << " * |V|\n";
  out << "  Isolated nodes:             " << lp_ctx.isolated_nodes_strategy << "\n";
}

void print(const InitialPartitioningContext &i_ctx, std::ostream &out) {
  out << "Adaptive algorithm selection: "
      << (i_ctx.use_adaptive_bipartitioner_selection ? "yes" : "no") << "\n";
}

void print(const RefinementContext &r_ctx, std::ostream &out) {
  out << "Refinement algorithms:        [" << str::implode(r_ctx.algorithms, " -> ") << "]\n";
  if (r_ctx.includes_algorithm(RefinementAlgorithm::LABEL_PROPAGATION)) {
    out << "Label propagation:\n";
    out << "  Number of iterations:       " << r_ctx.lp.num_iterations << "\n";
  }
  if (r_ctx.includes_algorithm(RefinementAlgorithm::KWAY_FM)) {
    out << "k-way FM:\n";
    out << "  Number of iterations:       " << r_ctx.kway_fm.num_iterations
        << " [or improvement drops below < " << 100.0 * (1.0 - r_ctx.kway_fm.abortion_threshold)
        << "%]\n";
    out << "  Number of seed nodes:       " << r_ctx.kway_fm.num_seed_nodes << "\n";
    out << "  Locking strategies:         seed nodes: "
        << (r_ctx.kway_fm.unlock_seed_nodes ? "unlock" : "lock") << ", locally moved nodes:"
        << (r_ctx.kway_fm.unlock_locally_moved_nodes ? "unlock" : "lock") << "\n";
    out << "  Gain cache:                 " << r_ctx.kway_fm.gain_cache_strategy << "\n";
    if (r_ctx.kway_fm.gain_cache_strategy == GainCacheStrategy::HYBRID) {
      out << "  High-degree threshold:\n";
      out << "    based on k:               " << r_ctx.kway_fm.k_based_high_degree_threshold
          << "\n";
      out << "    constant:                 " << r_ctx.kway_fm.constant_high_degree_threshold
          << "\n";
    }
  }
  if (r_ctx.includes_algorithm(RefinementAlgorithm::JET)) {
    out << "Jet refinement:               " << RefinementAlgorithm::JET << "\n";
    out << "  Number of iterations:       max " << r_ctx.jet.num_iterations << ", or "
        << r_ctx.jet.num_fruitless_iterations << " fruitless (improvement < "
        << 100.0 * (1 - r_ctx.jet.fruitless_threshold) << "%)\n";
    out << "  Penalty factors:            coarse " << r_ctx.jet.coarse_negative_gain_factor
        << ", fine " << r_ctx.jet.fine_negative_gain_factor << "\n";
    out << "  Balancing algorithm:        " << r_ctx.jet.balancing_algorithm << "\n";
  }
}

void print(const PartitionContext &p_ctx, std::ostream &out) {
  const auto max_block_weight = static_cast<std::int64_t>(p_ctx.block_weights.max(0));
  const auto size = std::max<std::int64_t>(
      {static_cast<std::int64_t>(p_ctx.n), static_cast<std::int64_t>(p_ctx.m), max_block_weight}
  );
  const std::size_t width = std::ceil(std::log10(size));

  out << "  Number of nodes:            " << std::setw(width) << p_ctx.n;
  if (asserting_cast<NodeWeight>(p_ctx.n) == p_ctx.total_node_weight) {
    out << " (unweighted)\n";
  } else {
    out << " (total weight: " << p_ctx.total_node_weight << ")\n";
  }
  out << "  Number of edges:            " << std::setw(width) << p_ctx.m;
  if (asserting_cast<EdgeWeight>(p_ctx.m) == p_ctx.total_edge_weight) {
    out << " (unweighted)\n";
  } else {
    out << " (total weight: " << p_ctx.total_edge_weight << ")\n";
  }
  out << "Number of blocks:             " << p_ctx.k << "\n";
  out << "Maximum block weight:         " << p_ctx.block_weights.max(0) << " ("
      << p_ctx.block_weights.perfectly_balanced(0) << " + " << 100 * p_ctx.epsilon << "%)\n";
}

void print(const PartitioningContext &p_ctx, std::ostream &out) {
  out << "Partitioning mode:            " << p_ctx.mode << "\n";
  if (p_ctx.mode == PartitioningMode::DEEP) {
    out << "  Deep initial part. mode:    " << p_ctx.deep_initial_partitioning_mode << "\n";
    out << "  Deep initial part. load:    " << p_ctx.deep_initial_partitioning_load << "\n";
  }
}

void print(const Context &ctx, std::ostream &out) {
  out << "Execution mode:               " << ctx.parallel.num_threads << "\n";
  out << "Seed:                         " << Random::get_seed() << "\n";
  out << "Graph:                        " << ctx.debug.graph_name
      << " [ordering: " << ctx.rearrange_by << "]\n";
  print(ctx.partition, out);
  cio::print_delimiter("Partitioning Scheme", '-');
  print(ctx.partitioning, out);
  cio::print_delimiter("Coarsening", '-');
  print(ctx.coarsening, out);
  cio::print_delimiter("Initial Partitioning", '-');
  print(ctx.initial_partitioning, out);
  cio::print_delimiter("Refinement", '-');
  print(ctx.refinement, out);
}
} // namespace kaminpar::shm
