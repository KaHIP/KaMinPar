/*******************************************************************************
 * @file:   context_io.cc
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 * @brief:  Input / output utility functions for the `shm::Context` struct.
 ******************************************************************************/
#include "kaminpar/context_io.h"

#include <algorithm>
#include <cmath>
#include <iomanip>

#include "common/asserting_cast.h"
#include "common/console_io.h"
#include "common/strutils.h"

namespace kaminpar::shm {
using namespace std::string_literals;

std::unordered_map<std::string, ClusteringAlgorithm>
get_clustering_algorithms() {
  return {
      {"noop", ClusteringAlgorithm::NOOP},
      {"lp", ClusteringAlgorithm::LABEL_PROPAGATION},
  };
}

std::ostream &
operator<<(std::ostream &out, const ClusteringAlgorithm algorithm) {
  switch (algorithm) {
  case ClusteringAlgorithm::NOOP:
    return out << "noop";
  case ClusteringAlgorithm::LABEL_PROPAGATION:
    return out << "lp";
  }
  return out << "<invalid>";
}

std::unordered_map<std::string, ClusterWeightLimit>
get_cluster_weight_limits() {
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

std::unordered_map<std::string, RefinementAlgorithm>
get_2way_refinement_algorithms() {
  return {
      {"noop", RefinementAlgorithm::NOOP},
      {"fm", RefinementAlgorithm::TWOWAY_FM},
  };
}

std::unordered_map<std::string, RefinementAlgorithm>
get_kway_refinement_algorithms() {
  return {
      {"noop", RefinementAlgorithm::NOOP},
      {"lp", RefinementAlgorithm::LABEL_PROPAGATION},
      {"fm", RefinementAlgorithm::KWAY_FM},
      {"greedy-balancer", RefinementAlgorithm::GREEDY_BALANCER},
  };
}

std::ostream &
operator<<(std::ostream &out, const RefinementAlgorithm algorithm) {
  switch (algorithm) {
  case RefinementAlgorithm::NOOP:
    return out << "noop";
  case RefinementAlgorithm::TWOWAY_FM:
  case RefinementAlgorithm::KWAY_FM:
    return out << "fm";
  case RefinementAlgorithm::LABEL_PROPAGATION:
    return out << "lp";
  case RefinementAlgorithm::GREEDY_BALANCER:
    return out << "greedy-balancer";
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
  };
}

std::ostream &operator<<(std::ostream &out, const PartitioningMode mode) {
  switch (mode) {
  case PartitioningMode::DEEP:
    return out << "deep";
  case PartitioningMode::RB:
    return out << "rb";
  }

  return out << "<invalid>";
}

std::unordered_map<std::string, InitialPartitioningMode>
get_initial_partitioning_modes() {
  return {
      {"sequential", InitialPartitioningMode::SEQUENTIAL},
      {"async-parallel", InitialPartitioningMode::ASYNCHRONOUS_PARALLEL},
      {"sync-parallel", InitialPartitioningMode::SYNCHRONOUS_PARALLEL},
  };
}

std::ostream &
operator<<(std::ostream &out, const InitialPartitioningMode mode) {
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
  out << "  High degree threshold:      " << lp_ctx.large_degree_threshold
      << "\n";
  out << "  Max degree:                 " << lp_ctx.max_num_neighbors << "\n";
  out << "  2-hop clustering threshold: " << std::fixed
      << 100 * lp_ctx.two_hop_clustering_threshold << "%\n";
}

void print(const InitialPartitioningContext &i_ctx, std::ostream &out) {
  out << "Initial partitioning mode:    " << i_ctx.mode << "\n";
  out << "Adaptive algorithm selection: "
      << (i_ctx.use_adaptive_bipartitioner_selection ? "yes" : "no") << "\n";
}

void print(const RefinementContext &r_ctx, std::ostream &out) {
  out << "Refinement algorithms:        ["
      << str::implode(r_ctx.algorithms, " -> ") << "]\n";
}

void print(const PartitionContext &p_ctx, std::ostream &out) {
  const std::int64_t max_block_weight = p_ctx.block_weights.max(0);
  const std::int64_t size =
      std::max<std::int64_t>({p_ctx.n, p_ctx.m, max_block_weight});
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
      << p_ctx.block_weights.perfectly_balanced(0) << " + "
      << 100 * p_ctx.epsilon << "%)\n";

  cio::print_delimiter("Partitioning Scheme", '-');
  out << "Partitioning mode:            " << p_ctx.mode << "\n";
}

void print(const Context &ctx, std::ostream &out) {
  out << "Execution mode:               " << ctx.parallel.num_threads << "\n";
  out << "Seed:                         " << ctx.seed << "\n";
  out << "Graph:                        " << ctx.graph_filename << "\n";
  print(ctx.partition, out);
  cio::print_delimiter("Coarsening", '-');
  print(ctx.coarsening, out);
  cio::print_delimiter("Initial Partitioning", '-');
  print(ctx.initial_partitioning, out);
  cio::print_delimiter("Refinement", '-');
  print(ctx.refinement, out);
}
} // namespace kaminpar::shm
