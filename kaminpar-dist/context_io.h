/*******************************************************************************
 * Utility functions to read/write parts of the partitioner context from/to
 * strings.
 *
 * @file:   context_io.h
 * @author: Daniel Seemaier
 * @date:   27.10.2022
 ******************************************************************************/
#pragma once

#include <ostream>
#include <unordered_map>

#include "kaminpar-dist/context.h"

namespace kaminpar::dist {

std::ostream &operator<<(std::ostream &out, PartitioningMode mode);
std::ostream &operator<<(std::ostream &out, ClusteringAlgorithm algorithm);
std::ostream &operator<<(std::ostream &out, InitialPartitioningAlgorithm algorithm);
std::ostream &operator<<(std::ostream &out, RefinementAlgorithm algorithm);
std::ostream &operator<<(std::ostream &out, LabelPropagationMoveExecutionStrategy strategy);
std::ostream &operator<<(std::ostream &out, GraphOrdering ordering);
std::ostream &operator<<(std::ostream &out, ClusterSizeStrategy strategy);
std::ostream &operator<<(std::ostream &out, ClusterStrategy strategy);
std::ostream &operator<<(std::ostream &out, GainCacheStrategy strategy);
std::ostream &operator<<(std::ostream &out, ActiveSetStrategy strategy);

std::unordered_map<std::string, PartitioningMode> get_partitioning_modes();
std::unordered_map<std::string, ClusteringAlgorithm> get_clustering_algorithms();
std::unordered_map<std::string, InitialPartitioningAlgorithm> get_initial_partitioning_algorithms();
std::unordered_map<std::string, RefinementAlgorithm> get_kway_refinement_algorithms();
std::unordered_map<std::string, RefinementAlgorithm> get_balancing_algorithms();
std::unordered_map<std::string, LabelPropagationMoveExecutionStrategy>
get_label_propagation_move_execution_strategies();
std::unordered_map<std::string, GraphOrdering> get_graph_orderings();
std::unordered_map<std::string, GraphDistribution> get_graph_distributions();
std::unordered_map<std::string, ClusterSizeStrategy> get_move_set_size_strategies();
std::unordered_map<std::string, ClusterStrategy> get_move_set_strategies();
std::unordered_map<std::string, GainCacheStrategy> get_gain_cache_strategies();
std::unordered_map<std::string, ActiveSetStrategy> get_active_set_strategies();

std::string get_refinement_algorithms_description();
std::string get_balancing_algorithms_description();

void print(const Context &ctx, bool root, std::ostream &out, MPI_Comm comm);
void print(const PartitionContext &ctx, bool root, std::ostream &out, MPI_Comm comm);
void print(const ChunksContext &ctx, const ParallelContext &parallel, std::ostream &out);
void print(
    const GraphCompressionContext &ctx,
    const ParallelContext &parallel,
    const bool print_compression_details,
    std::ostream &out
);
void print(const CoarseningContext &ctx, const ParallelContext &parallel, std::ostream &out);
void print(const InitialPartitioningContext &ctx, std::ostream &out);
void print(const RefinementContext &ctx, const ParallelContext &parallel, std::ostream &out);

template <typename T> std::string stringify_enum(const T val) {
  std::stringstream ss;
  ss << val;
  return ss.str();
}

} // namespace kaminpar::dist
