/*******************************************************************************
 * k-way multilevel graph partitioning scheme.
 *
 * @file:   kway_multilevel.h
 * @author: Daniel Seemaier
 * @date:   19.09.2023
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/coarsener.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/initial_partitioning/initial_bipartitioner_worker_pool.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/partitioning/partitioner.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {

class KWayMultilevelPartitioner : public Partitioner {
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

  std::unique_ptr<Coarsener> _coarsener;
  std::unique_ptr<Refiner> _refiner;

  InitialBipartitionerWorkerPool _bipartitioner_pool;
};

} // namespace kaminpar::shm
