/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/distributed_definitions.h"
#include "kaminpar/context.h"

namespace dkaminpar {
enum class PartitioningMode {
  KWAY,
  RB,
  DEEP,
};

enum class CoarseningAlgorithm {
  NOOP,
  LOCAL_LP,
};

enum class InitialPartitioningAlgorithm {
  KAMINPAR,
};

enum class KWayRefinementAlgorithm {
  NOOP,
  LP,
};

DECLARE_ENUM_STRING_CONVERSION(PartitioningMode, partitioning_mode);
DECLARE_ENUM_STRING_CONVERSION(CoarseningAlgorithm, coarsening_algorithm);
DECLARE_ENUM_STRING_CONVERSION(InitialPartitioningAlgorithm, initial_partitioning_algorithm);
DECLARE_ENUM_STRING_CONVERSION(KWayRefinementAlgorithm, kway_refinement_algorithm);

struct LabelPropagationCoarseningContext {
  std::size_t num_iterations;
  NodeID large_degree_threshold;
  NodeID max_num_neighbors;
  bool merge_singleton_clusters;
  double merge_nonadjacent_clusters_threshold;

  [[nodiscard]] bool should_merge_nonadjacent_clusters(const NodeID old_n, const NodeID new_n) const {
    return (1.0 - 1.0 * static_cast<double>(new_n) / static_cast<double>(old_n)) <=
           merge_nonadjacent_clusters_threshold;
  }

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct LabelPropagationRefinementContext {
  std::size_t num_iterations;
  std::size_t num_chunks;
  std::size_t num_move_attempts;

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct CoarseningContext {
  CoarseningAlgorithm algorithm;
  LabelPropagationCoarseningContext lp;

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct InitialPartitioningContext {
  InitialPartitioningAlgorithm algorithm;
  shm::Context sequential;

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct RefinementContext {
  KWayRefinementAlgorithm algorithm;
  LabelPropagationRefinementContext lp;

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct ParallelContext {
  std::size_t num_threads;
  bool use_interleaved_numa_allocation;
  int mpi_thread_support;

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct PartitionContext {
  BlockID k{};
  double epsilon{};
  PartitioningMode mode{};

  GlobalNodeID global_n{kInvalidGlobalNodeID};
  GlobalEdgeID global_m{kInvalidGlobalEdgeID};
  NodeID local_n{kInvalidNodeID};
  EdgeID local_m{kInvalidEdgeID};

  void setup(const DistributedGraph &graph);

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct Context {
  std::string graph_filename{};
  int seed{0};
  bool quiet{};

  PartitionContext partition;
  ParallelContext parallel;
  CoarseningContext coarsening;
  InitialPartitioningContext initial_partitioning;
  RefinementContext refinement;

  void setup(const DistributedGraph &graph) {
    partition.local_n = graph.n();
    partition.local_m = graph.m();
    partition.global_m = graph.global_n();
    partition.global_n = graph.global_m();
  }

  void print(std::ostream &out, const std::string &prefix = "") const;
};

std::ostream &operator<<(std::ostream &out, const Context &context);

Context create_default_context();
} // namespace dkaminpar