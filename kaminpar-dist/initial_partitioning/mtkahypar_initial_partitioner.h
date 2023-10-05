/*******************************************************************************
 * Initial partitioner that uses Mt-KaHypar. Only available if the Mt-KaHyPar
 * library is installed on the system.
 *
 * @file:   mtkahypar_initial_partitioner.h
 * @author: Daniel Seemaier
 * @date:   15.09.2022
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/initial_partitioning/initial_partitioner.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"

namespace kaminpar::dist {
class MtKaHyParInitialPartitioner : public InitialPartitioner {
public:
  MtKaHyParInitialPartitioner(const Context &ctx) : _ctx{ctx} {}

  MtKaHyParInitialPartitioner(const MtKaHyParInitialPartitioner &) = delete;
  MtKaHyParInitialPartitioner &operator=(const MtKaHyParInitialPartitioner &) = delete;
  MtKaHyParInitialPartitioner(MtKaHyParInitialPartitioner &&) noexcept = default;
  MtKaHyParInitialPartitioner &operator=(MtKaHyParInitialPartitioner &&) = delete;

  shm::PartitionedGraph
  initial_partition(const shm::Graph &graph, const PartitionContext &p_ctx) override;

private:
  const Context &_ctx;
};
} // namespace kaminpar::dist
