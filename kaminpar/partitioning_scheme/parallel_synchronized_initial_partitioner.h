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

#include "coarsening/parallel_label_propagation_coarsener.h"
#include "datastructure/graph.h"
#include "helper.h"
#include "partitioning_scheme/helper.h"
#include "refinement/parallel_balancer.h"
#include "refinement/parallel_label_propagation_refiner.h"

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_invoke.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#include <tbb/task_scheduler_observer.h>

namespace kaminpar::partitioning {
class ParallelSynchronizedInitialPartitioner {
  SET_DEBUG(false);

public:
  ParallelSynchronizedInitialPartitioner(const Context &input_ctx, GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool,
                                         TemporaryGraphExtractionBufferPool &ip_extraction_pool);

  PartitionedGraph partition(const Coarsener *coarsener, const PartitionContext &p_ctx);

private:
  std::unique_ptr<Coarsener> duplicate_coarsener(const Coarsener *coarsener);

  const Context &_input_ctx;
  GlobalInitialPartitionerMemoryPool &_ip_m_ctx_pool;
  TemporaryGraphExtractionBufferPool &_ip_extraction_pool;
};
} // namespace kaminpar::partitioning
