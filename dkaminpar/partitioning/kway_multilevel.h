/*******************************************************************************
 * Multilevel graph partitioning with direct k-way initial partitioning.
 *
 * @file:   kway_partitioner.h
 * @author: Daniel Seemaier
 * @date:   25.10.2021
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/partitioning/partitioner.h"

namespace kaminpar::dist {
class KWayMultilevelPartitioner : public Partitioner {
public:
  KWayMultilevelPartitioner(const DistributedGraph &graph, const Context &ctx);

  KWayMultilevelPartitioner(const KWayMultilevelPartitioner &) = delete;
  KWayMultilevelPartitioner &operator=(const KWayMultilevelPartitioner &) = delete;
  KWayMultilevelPartitioner(KWayMultilevelPartitioner &&) noexcept = default;
  KWayMultilevelPartitioner &operator=(KWayMultilevelPartitioner &&) = delete;

  DistributedPartitionedGraph partition() final;

private:
  const DistributedGraph &_graph;
  const Context &_ctx;
};
} // namespace kaminpar::dist
