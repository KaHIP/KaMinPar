/*******************************************************************************
 * Implementation of the n / 2C < P phase of deep multilevel graph partitioning
 * scheduling the PE groups asynchronously.
 *
 * @file:   async_initial_partitioning.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/coarsener.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/initial_partitioning/initial_bipartitioner_worker_pool.h"
#include "kaminpar-shm/partitioning/helper.h"

namespace kaminpar::shm::partitioning {

class AsyncInitialPartitioner {
public:
  AsyncInitialPartitioner(
      const Context &input_ctx,
      InitialBipartitionerWorkerPool &bipartitioner_pool,
      TemporarySubgraphMemoryEts &tmp_extraction_mem_pool_ets
  );

  PartitionedGraph partition(const Coarsener *coarsener, const PartitionContext &p_ctx);

private:
  PartitionedGraph partition_recursive(
      const Coarsener *parent_coarsener, PartitionContext &p_ctx, std::size_t num_threads
  );

  PartitionedGraph split_and_join(
      const Coarsener *coarsener,
      const PartitionContext &p_ctx,
      bool converged,
      std::size_t num_threads
  );

  const Context &_input_ctx;

  InitialBipartitionerWorkerPool &_bipartitioner_pool;
  TemporarySubgraphMemoryEts &_tmp_extraction_mem_pool_ets;
};

} // namespace kaminpar::shm::partitioning
