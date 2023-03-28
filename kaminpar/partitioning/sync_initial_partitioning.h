/*******************************************************************************
 * @file:   parallel_synchronized_initial_partitioner.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:
 ******************************************************************************/
#pragma once

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_invoke.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#include <tbb/task_scheduler_observer.h>

#include "kaminpar/coarsening/label_propagation_clustering.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/partitioning/helper.h"
#include "kaminpar/refinement/greedy_balancer.h"
#include "kaminpar/refinement/label_propagation_refiner.h"

namespace kaminpar::shm::partitioning {
class SyncInitialPartitioner {
  SET_DEBUG(false);

public:
  SyncInitialPartitioner(
      const Context &input_ctx,
      GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool,
      TemporaryGraphExtractionBufferPool &ip_extraction_pool);

  PartitionedGraph partition(const Coarsener *coarsener,
                             const PartitionContext &p_ctx);

private:
  std::unique_ptr<Coarsener> duplicate_coarsener(const Coarsener *coarsener);

  const Context &_input_ctx;
  GlobalInitialPartitionerMemoryPool &_ip_m_ctx_pool;
  TemporaryGraphExtractionBufferPool &_ip_extraction_pool;
};
} // namespace kaminpar::shm::partitioning
