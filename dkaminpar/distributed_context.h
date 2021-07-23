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
enum PartitioningMode {
  KWAY,
  RB,
  DEEP,
};

DECLARE_ENUM_STRING_CONVERSION(PartitioningMode, partitioning_mode);

struct DLabelPropagationCoarseningContext {
  std::size_t num_iterations;
  DNodeID large_degree_threshold;
  DNodeID max_num_neighbors;
  bool merge_singleton_clusters;
  double merge_nonadjacent_clusters_threshold;

  [[nodiscard]] bool should_merge_nonadjacent_clusters(const DNodeID old_n, const DNodeID new_n) const {
    return (1.0 - 1.0 * new_n / old_n) <= merge_nonadjacent_clusters_threshold;
  }

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct DLabelPropagationRefinementContext {
  std::size_t num_iterations;
  std::size_t num_chunks;
  std::size_t num_move_attempts;

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct DCoarseningContext {
  DLabelPropagationCoarseningContext lp;

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct DInitialPartitioning {
  shm::Context sequential;

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct DRefinementContext {
  DLabelPropagationRefinementContext lp;

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct DParallelContext {
  std::size_t num_threads;
  bool use_interleaved_numa_allocation;

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct DPartitionContext {
  DBlockID k{};
  double epsilon{};
  PartitioningMode mode{};

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct DContext {
  std::string graph_filename{};
  int seed{0};

  DPartitionContext partition;
  DParallelContext parallel;
  DCoarseningContext coarsening;
  DInitialPartitioning initial_partitioning;
  DRefinementContext refinement;

  void setup(const DistributedGraph &graph) {
    UNUSED(graph);
  }

  void print(std::ostream &out, const std::string &prefix = "") const;
};

DContext create_default_context();
} // namespace dkaminpar