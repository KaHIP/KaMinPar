/*******************************************************************************
 * @file:   context_io.h
 * @author: Daniel Seemaier
 * @date:   27.10.2022
 * @brief:  Utility functions to read/write parts of the partitioner context
 * from/to strings.
 ******************************************************************************/
#pragma once

#include <ostream>
#include <unordered_map>

#include "dkaminpar/context.h"

namespace kaminpar::dist {
std::ostream& operator<<(std::ostream& out, PartitioningMode mode);
std::ostream& operator<<(std::ostream& out, GlobalClusteringAlgorithm algorithm);
std::ostream& operator<<(std::ostream& out, LocalClusteringAlgorithm algorithm);
std::ostream& operator<<(std::ostream& out, GlobalContractionAlgorithm algorithm);
std::ostream& operator<<(std::ostream& out, InitialPartitioningAlgorithm algorithm);
std::ostream& operator<<(std::ostream& out, KWayRefinementAlgorithm algorithm);

std::unordered_map<std::string, PartitioningMode>             get_partitioning_modes();
std::unordered_map<std::string, GlobalClusteringAlgorithm>    get_global_clustering_algorithms();
std::unordered_map<std::string, LocalClusteringAlgorithm>     get_local_clustering_algorithms();
std::unordered_map<std::string, GlobalContractionAlgorithm>   get_global_contraction_algorithms();
std::unordered_map<std::string, InitialPartitioningAlgorithm> get_initial_partitioning_algorithms();
std::unordered_map<std::string, KWayRefinementAlgorithm>      get_kway_refinement_algorithms();

void print_compact(const Context& ctx, std::ostream& out, const std::string& prefix = "");
void print_compact(const DebugContext& ctx, std::ostream& out, const std::string& prefix);
void print_compact(const PartitionContext& ctx, std::ostream& out, const std::string& prefix);
void print_compact(const ParallelContext& ctx, std::ostream& out, const std::string& prefix);
void print_compact(const RefinementContext& ctx, std::ostream& out, const std::string& prefix);
void print_compact(const InitialPartitioningContext& ctx, std::ostream& out, const std::string& prefix);
void print_compact(const MtKaHyParContext& ctx, std::ostream& out, const std::string& prefix);
void print_compact(const GreedyBalancerContext& ctx, std::ostream& out, const std::string& prefix);
void print_compact(const CoarseningContext& ctx, std::ostream& out, const std::string& prefix);
void print_compact(const FMRefinementContext& ctx, std::ostream& out, const std::string& prefix);
void print_compact(const LabelPropagationRefinementContext& ctx, std::ostream& out, const std::string& prefix);
void print_compact(const LabelPropagationCoarseningContext& ctx, std::ostream& out, const std::string& prefix);

void print(const Context& ctx, bool root, std::ostream& out);
void print(const PartitionContext& ctx, bool root, std::ostream& out);
void print(const CoarseningContext& ctx, std::ostream& out);
void print(const InitialPartitioningContext& ctx, std::ostream& out);
void print(const RefinementContext& ctx, std::ostream& out);
} // namespace kaminpar::dist
