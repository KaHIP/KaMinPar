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

#include "algorithm/graph_utils.h"
#include "coarsening/parallel_label_propagation_coarsener.h"
#include "context.h"
#include "datastructure/graph.h"
#include "factories.h"
#include "initial_partitioning/initial_partitioning_facade.h"
#include "initial_partitioning/pool_bipartitioner.h"
#include "partitioning_scheme/helper.h"
#include "utility/console_io.h"

#include <tbb/enumerable_thread_specific.h>

namespace kaminpar::partitioning {
class ParallelRecursiveBisection {
  static constexpr bool kDebug = false;
  static constexpr bool kStatistics = false;

public:
  ParallelRecursiveBisection(const Graph &input_graph, const Context &input_ctx);

  ParallelRecursiveBisection(const ParallelRecursiveBisection &) = delete;
  ParallelRecursiveBisection &operator=(const ParallelRecursiveBisection &) = delete;
  ParallelRecursiveBisection(ParallelRecursiveBisection &&) = delete;
  ParallelRecursiveBisection &operator=(ParallelRecursiveBisection &&) = delete;

  PartitionedGraph partition();

private:
  PartitionedGraph uncoarsen(PartitionedGraph p_graph, bool &refined);

  inline PartitionedGraph uncoarsen_once(PartitionedGraph p_graph) {
    return helper::uncoarsen_once(_coarsener.get(), std::move(p_graph), _current_p_ctx);
  }

  inline void refine(PartitionedGraph &p_graph) {
    helper::refine(_refiner.get(), _balancer.get(), p_graph, _current_p_ctx, _input_ctx.refinement);
  }

  inline void extend_partition(PartitionedGraph &p_graph, const BlockID k_prime) {
    helper::extend_partition(p_graph, k_prime, _input_ctx, _current_p_ctx, _subgraph_memory, _ip_extraction_pool,
                             _ip_m_ctx_pool);
  }

  const Graph *coarsen();
  NodeID initial_partition_threshold();
  PartitionedGraph initial_partition(const Graph *graph);
  PartitionedGraph parallel_initial_partition(const Graph * /* use _coarsener */);
  PartitionedGraph sequential_initial_partition(const Graph *graph);
  void print_statistics();

  const Graph &_input_graph;
  const Context &_input_ctx;
  PartitionContext _current_p_ctx;

  // Coarsening
  std::unique_ptr<Coarsener> _coarsener;

  // Refinement
  std::unique_ptr<Refiner> _refiner;
  std::unique_ptr<Balancer> _balancer;

  // Initial partitioning -> subgraph extraction
  SubgraphMemory _subgraph_memory;
  TemporaryGraphExtractionBufferPool _ip_extraction_pool;

  // Initial partitioning
  GlobalInitialPartitionerMemoryPool _ip_m_ctx_pool;
};
} // namespace kaminpar::partitioning