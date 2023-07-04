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

#include "dkaminpar/context.h"

namespace kaminpar::dist {
std::ostream &operator<<(std::ostream &out, PartitioningMode mode);
std::ostream &operator<<(std::ostream &out, GlobalClusteringAlgorithm algorithm);
std::ostream &operator<<(std::ostream &out, LocalClusteringAlgorithm algorithm);
std::ostream &operator<<(std::ostream &out, InitialPartitioningAlgorithm algorithm);
std::ostream &operator<<(std::ostream &out, KWayRefinementAlgorithm algorithm);
std::ostream &operator<<(std::ostream &out, LabelPropagationMoveExecutionStrategy strategy);
std::ostream &operator<<(std::ostream &out, GraphOrdering ordering);

std::unordered_map<std::string, PartitioningMode> get_partitioning_modes();
std::unordered_map<std::string, GlobalClusteringAlgorithm> get_global_clustering_algorithms();
std::unordered_map<std::string, LocalClusteringAlgorithm> get_local_clustering_algorithms();
std::unordered_map<std::string, InitialPartitioningAlgorithm> get_initial_partitioning_algorithms();
std::unordered_map<std::string, KWayRefinementAlgorithm> get_kway_refinement_algorithms();
std::unordered_map<std::string, LabelPropagationMoveExecutionStrategy>
get_label_propagation_move_execution_strategies();
std::unordered_map<std::string, GraphOrdering> get_graph_orderings();

void print(const Context &ctx, bool root, std::ostream &out);
void print(const PartitionContext &ctx, bool root, std::ostream &out);
void print(const CoarseningContext &ctx, const ParallelContext &parallel, std::ostream &out);
void print(const InitialPartitioningContext &ctx, std::ostream &out);
void print(const RefinementContext &ctx, std::ostream &out);
} // namespace kaminpar::dist
