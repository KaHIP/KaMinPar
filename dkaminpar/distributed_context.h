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
  std::size_t num_chunks;

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
  // required for braces-initializer with private members
  PartitionContext(const BlockID k, const double epsilon, const PartitioningMode mode)
      : k{k},
        epsilon{epsilon},
        mode{mode} {}

  BlockID k{};
  double epsilon{};
  PartitioningMode mode{};

  void setup(const DistributedGraph &graph);

  [[nodiscard]] GlobalNodeID global_n() const {
    ASSERT(_global_n != kInvalidGlobalNodeID);
    return _global_n;
  }

  [[nodiscard]] GlobalEdgeID global_m() const {
    ASSERT(_global_m != kInvalidGlobalEdgeID);
    return _global_m;
  }

  [[nodiscard]] GlobalNodeWeight global_total_node_weight() const {
    ASSERT(_global_total_node_weight != kInvalidGlobalNodeWeight);
    return _global_total_node_weight;
  }

  [[nodiscard]] NodeID local_n() const {
    ASSERT(_local_n != kInvalidNodeID);
    return _local_n;
  }

  [[nodiscard]] EdgeID local_m() const {
    ASSERT(_local_m != kInvalidEdgeID);
    return _local_m;
  }

  [[nodiscard]] NodeWeight total_node_weight() const {
    ASSERT(_total_node_weight != kInvalidNodeWeight);
    return _total_node_weight;
  }

  [[nodiscard]] inline BlockWeight perfectly_balanced_block_weight(const BlockID b) const {
    ASSERT(b < _perfectly_balanced_block_weights.size());
    return _perfectly_balanced_block_weights[b];
  }

  [[nodiscard]] inline BlockWeight max_block_weight(const BlockID b) const {
    ASSERT(b < _max_block_weights.size());
    return _max_block_weights[b];
  }

  void print(std::ostream &out, const std::string &prefix = "") const;

private:
  void setup_perfectly_balanced_block_weights();
  void setup_max_block_weights();

  GlobalNodeID _global_n{kInvalidGlobalNodeID};
  GlobalEdgeID _global_m{kInvalidGlobalEdgeID};
  GlobalNodeWeight _global_total_node_weight{kInvalidGlobalNodeWeight};
  NodeID _local_n{kInvalidNodeID};
  EdgeID _local_m{kInvalidEdgeID};
  NodeWeight _total_node_weight{kInvalidNodeWeight};

  scalable_vector<BlockWeight> _perfectly_balanced_block_weights{};
  scalable_vector<BlockWeight> _max_block_weights{};
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

  void setup(const DistributedGraph &graph) { partition.setup(graph); }

  void print(std::ostream &out, const std::string &prefix = "") const;
};

std::ostream &operator<<(std::ostream &out, const Context &context);

Context create_default_context();
} // namespace dkaminpar