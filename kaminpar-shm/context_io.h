/*******************************************************************************
 * IO functions for the context structs.
 *
 * @file:   context_io.h
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 ******************************************************************************/
#pragma once

#include <iostream>
#include <unordered_map>

#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {
std::ostream &operator<<(std::ostream &out, GraphOrdering ordering);

std::unordered_map<std::string, GraphOrdering> get_graph_orderings();

std::ostream &operator<<(std::ostream &out, ClusteringAlgorithm algorithm);

std::unordered_map<std::string, ClusteringAlgorithm> get_clustering_algorithms();

std::ostream &operator<<(std::ostream &out, ClusterWeightLimit limit);

std::unordered_map<std::string, ClusterWeightLimit> get_cluster_weight_limits();

std::ostream &operator<<(std::ostream &out, RefinementAlgorithm algorithm);

std::unordered_map<std::string, RefinementAlgorithm> get_kway_refinement_algorithms();

std::ostream &operator<<(std::ostream &out, FMStoppingRule rule);

std::unordered_map<std::string, FMStoppingRule> get_fm_stopping_rules();

std::ostream &operator<<(std::ostream &out, PartitioningMode mode);

std::unordered_map<std::string, PartitioningMode> get_partitioning_modes();

std::ostream &operator<<(std::ostream &out, InitialPartitioningMode mode);

std::unordered_map<std::string, InitialPartitioningMode> get_initial_partitioning_modes();

std::ostream &operator<<(std::ostream &out, GainCacheStrategy strategy);

std::unordered_map<std::string, GainCacheStrategy> get_gain_cache_strategies();

std::ostream &operator<<(std::ostream &out, TwoHopStrategy strategy);

std::unordered_map<std::string, TwoHopStrategy> get_two_hop_strategies();

std::ostream &operator<<(std::ostream &out, IsolatedNodesClusteringStrategy strategy);

std::unordered_map<std::string, IsolatedNodesClusteringStrategy> get_isolated_nodes_clustering_strategies();

void print(const Context &ctx, std::ostream &out);
void print(const PartitioningContext &p_ctx, std::ostream &out);
void print(const PartitionContext &p_ctx, std::ostream &out);
void print(const RefinementContext &r_ctx, std::ostream &out);
void print(const CoarseningContext &c_ctx, std::ostream &out);
void print(const LabelPropagationCoarseningContext &lp_ctx, std::ostream &out);
void print(const InitialPartitioningContext &i_ctx, std::ostream &out);
} // namespace kaminpar::shm
