/*******************************************************************************
 * Partitioning scheme that uses toplevel multilevel recursvie bipartitioning.
 *
 * @file:   rb_multilevel.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <tbb/parallel_invoke.h>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/initial_partitioning/initial_bipartitioner_worker_pool.h"
#include "kaminpar-shm/partitioning/partitioner.h"

namespace kaminpar::shm {

class RBMultilevelPartitioner : public Partitioner {
public:
  RBMultilevelPartitioner(const Graph &graph, const Context &ctx);

  RBMultilevelPartitioner(const RBMultilevelPartitioner &) = delete;
  RBMultilevelPartitioner &operator=(const RBMultilevelPartitioner &) = delete;

  RBMultilevelPartitioner(RBMultilevelPartitioner &&) = delete;
  RBMultilevelPartitioner &operator=(RBMultilevelPartitioner &&) = delete;

  PartitionedGraph partition() final;

private:
  PartitionedGraph partition_recursive(const Graph &graph, BlockID k);

  PartitionedGraph bipartition(const Graph &graph, BlockID final_k);

  const Graph &_input_graph;
  const Context &_input_ctx;

  InitialBipartitionerWorkerPool _bipartitioner_pool;
};

} // namespace kaminpar::shm
