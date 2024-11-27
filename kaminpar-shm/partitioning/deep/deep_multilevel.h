/*******************************************************************************
 * Deep multilevel graph partitioning scheme.
 *
 * @file:   deep_multilevel.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <span>

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-shm/coarsening/coarsener.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/graphutils/subgraph_extractor.h"
#include "kaminpar-shm/initial_partitioning/initial_bipartitioner_worker_pool.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/partitioning/helper.h"
#include "kaminpar-shm/partitioning/partitioner.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm {

class DeepMultilevelPartitioner : public Partitioner {
public:
  DeepMultilevelPartitioner(const Graph &input_graph, const Context &input_ctx);

  DeepMultilevelPartitioner(const DeepMultilevelPartitioner &) = delete;
  DeepMultilevelPartitioner &operator=(const DeepMultilevelPartitioner &) = delete;

  DeepMultilevelPartitioner(DeepMultilevelPartitioner &&) = delete;
  DeepMultilevelPartitioner &operator=(DeepMultilevelPartitioner &&) = delete;

  void use_communities(std::span<const NodeID> communities, NodeID num_communities);

  PartitionedGraph partition() final;

private:
  const Graph *coarsen();

  NodeID initial_partitioning_threshold();

  PartitionedGraph initial_partition(const Graph *graph);

  StaticArray<BlockID> copy_coarsest_communities();
  PartitionedGraph initial_partition_by_communities(const Graph *graph);

  PartitionedGraph uncoarsen(PartitionedGraph p_graph);

  void refine(PartitionedGraph &p_graph);

  inline void extend_partition(PartitionedGraph &p_graph, BlockID k_prime);

  void print_statistics();

  const Graph &_input_graph;
  const Context &_input_ctx;

  PartitionContext _current_p_ctx;

  std::unique_ptr<Coarsener> _coarsener = nullptr;
  std::unique_ptr<Refiner> _refiner = nullptr;

  std::size_t _last_initial_partitioning_level = 0;
  NodeID _subgraph_memory_n = 0;
  NodeID _subgraph_memory_n_weights = 0;
  EdgeID _subgraph_memory_m = 0;
  EdgeID _subgraph_memory_m_weights = 0;

  graph::SubgraphMemory _subgraph_memory;
  partitioning::SubgraphMemoryEts _extraction_mem_pool_ets;
  partitioning::TemporarySubgraphMemoryEts _tmp_extraction_mem_pool_ets;
  InitialBipartitionerWorkerPool _bipartitioner_pool;

  NodeID _num_communities = 0;
};

} // namespace kaminpar::shm
