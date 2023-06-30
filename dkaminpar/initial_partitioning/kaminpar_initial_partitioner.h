/*******************************************************************************
 * Initial partitioner invoking KaMinPar.
 *
 * @file:   kaminpar_initial_partitioner.h
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/initial_partitioning/initial_partitioner.h"

#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"

namespace kaminpar::dist {
class KaMinParInitialPartitioner : public InitialPartitioner {
public:
  KaMinParInitialPartitioner(const Context &ctx) : _ctx{ctx} {}

  KaMinParInitialPartitioner(const KaMinParInitialPartitioner &) = delete;
  KaMinParInitialPartitioner &operator=(const KaMinParInitialPartitioner &) = delete;
  KaMinParInitialPartitioner(KaMinParInitialPartitioner &&) noexcept = default;
  KaMinParInitialPartitioner &operator=(KaMinParInitialPartitioner &&) = delete;

  shm::PartitionedGraph
  initial_partition(const shm::Graph &graph, const PartitionContext &p_ctx) override;

private:
  const Context &_ctx;
};
} // namespace kaminpar::dist
