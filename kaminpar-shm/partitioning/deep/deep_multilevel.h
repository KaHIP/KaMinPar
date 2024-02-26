/*******************************************************************************
 * Deep multilevel graph partitioning scheme.
 *
 * @file:   deep_multilevel.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/graphutils/subgraph_extractor.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/partitioning/helper.h"
#include "kaminpar-shm/partitioning/partitioner.h"

namespace kaminpar::shm {
class DeepMultilevelPartitioner : public Partitioner {
public:
  DeepMultilevelPartitioner(const Graph &input_graph, const Context &input_ctx);

  DeepMultilevelPartitioner(const DeepMultilevelPartitioner &) = delete;
  DeepMultilevelPartitioner &operator=(const DeepMultilevelPartitioner &) = delete;

  DeepMultilevelPartitioner(DeepMultilevelPartitioner &&) = delete;
  DeepMultilevelPartitioner &operator=(DeepMultilevelPartitioner &&) = delete;

  PartitionedGraph partition() final;

private:
  PartitionedGraph uncoarsen(PartitionedGraph p_graph, bool &refined);

  inline PartitionedGraph uncoarsen_once(PartitionedGraph p_graph);

  void refine(PartitionedGraph &p_graph);

  inline void extend_partition(PartitionedGraph &p_graph, BlockID k_prime);

  const Graph *coarsen();

  NodeID initial_partitioning_threshold();

  PartitionedGraph initial_partition(const Graph *graph);

  void print_statistics();

  const Graph &_input_graph;
  const Context &_input_ctx;
  PartitionContext _current_p_ctx;

  // Coarsening
  std::unique_ptr<Coarsener> _coarsener;

  // Refinement
  std::unique_ptr<Refiner> _refiner;

  // Initial partitioning -> subgraph extraction
  graph::SubgraphMemory _subgraph_memory;
  partitioning::TemporaryGraphExtractionBufferPool _ip_extraction_pool;

  // Initial partitioning
  partitioning::GlobalInitialPartitionerMemoryPool _ip_m_ctx_pool;
};
} // namespace kaminpar::shm
