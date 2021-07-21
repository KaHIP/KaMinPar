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

namespace dkaminpar {
struct DLabelPropagationCoarseningContext {
  std::size_t num_iterations;
  DNodeID large_degree_threshold;
  DNodeID max_num_neighbors;
  bool merge_singleton_clusters;
  double merge_nonadjacent_clusters_threshold;

  [[nodiscard]] bool should_merge_nonadjacent_clusters(const DNodeID old_n, const DNodeID new_n) const {
    return (1.0 - 1.0 * new_n / old_n) <= merge_nonadjacent_clusters_threshold;
  }
};

struct DLabelPropagationRefinementContext {
  std::size_t num_iterations;
  std::size_t num_chunks;
  std::size_t num_move_attempts;
};

struct DCoarseningContext {
  DLabelPropagationCoarseningContext lp;
};

struct DRefinementContext {
  DLabelPropagationRefinementContext lp;
};

struct DParallelContext {
  std::size_t num_threads;
  bool use_interleaved_numa_allocation;
};

struct DPartitionContext {
  DBlockID k{};
  double epsilon{};
};

struct DContext {
  std::string graph_filename{};
  int seed{0};

  DPartitionContext partition;
  DParallelContext parallel;
  DCoarseningContext coarsening;
  DRefinementContext refinement;

  void setup(const DistributedGraph &graph) {
    UNUSED(graph);
  }
};

DContext create_default_context() {
  // clang-format off
  return {
    .graph_filename = "",
    .seed = 0,
    .partition = {
      .k = 0,
      .epsilon = 0.03,
    },
    .parallel = {
      .num_threads = 1,
      .use_interleaved_numa_allocation = true,
    },
    .coarsening = {
      .lp = {
        .num_iterations = 5,
        .large_degree_threshold = 1000000,
        .max_num_neighbors = std::numeric_limits<DNodeID>::max(),
        .merge_singleton_clusters = true,
        .merge_nonadjacent_clusters_threshold = 0.5,
      }
    },
    .refinement = {
      .lp = {
        .num_iterations = 5,
        .num_chunks = 8,
        .num_move_attempts = 2,
      }
    }
  };
  // clang-format on
}
} // namespace dkaminpar