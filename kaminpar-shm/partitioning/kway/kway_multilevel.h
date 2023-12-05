/*******************************************************************************
 * k-way multilevel graph partitioning scheme.
 *
 * @file:   kway_multilevel.h
 * @author: Daniel Seemaier
 * @date:   19.09.2023
 ******************************************************************************/
#pragma once

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/factories.h"
#include "kaminpar-shm/graphutils/subgraph_extractor.h"
#include "kaminpar-shm/partitioning/helper.h"
#include "kaminpar-shm/partitioning/partitioner.h"

namespace kaminpar::shm {
class KWayMultilevelPartitioner : public Partitioner {
  SET_DEBUG(false);
  SET_STATISTICS(false);

public:
  KWayMultilevelPartitioner(const Graph &input_graph, const Context &input_ctx);

  KWayMultilevelPartitioner(const KWayMultilevelPartitioner &) = delete;
  KWayMultilevelPartitioner &operator=(const KWayMultilevelPartitioner &) = delete;

  KWayMultilevelPartitioner(KWayMultilevelPartitioner &&) = delete;
  KWayMultilevelPartitioner &operator=(KWayMultilevelPartitioner &&) = delete;

  PartitionedGraph partition() final;

private:
  PartitionedGraph uncoarsen(PartitionedGraph p_graph);

  void refine(PartitionedGraph &p_graph);

  const Graph *coarsen();

  NodeID initial_partitioning_threshold();

  PartitionedGraph initial_partition(const Graph *graph);

  const Graph &_input_graph;
  const Context &_input_ctx;
  PartitionContext _current_p_ctx;

  // Coarsening
  std::unique_ptr<Coarsener> _coarsener;

  // Refinement
  std::unique_ptr<Refiner> _refiner;

  // Initial partitioning
  partitioning::GlobalInitialPartitionerMemoryPool _ip_m_ctx_pool;
};
} // namespace kaminpar::shm
