/*******************************************************************************
 * @file:   kway_partitioner.h
 * @author: Daniel Seemaier
 * @date:   25.10.2021
 * @brief:  Multilevel graph partitioning with direct k-way initial
 *partitioning.
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"
#include "dkaminpar/partitioning/partitioner.h"

namespace kaminpar::dist {
class KWayPartitioner : public Partitioner {
public:
  KWayPartitioner(const DistributedGraph &graph, const Context &ctx);

  KWayPartitioner(const KWayPartitioner &) = delete;
  KWayPartitioner &operator=(const KWayPartitioner &) = delete;
  KWayPartitioner(KWayPartitioner &&) noexcept = default;
  KWayPartitioner &operator=(KWayPartitioner &&) = delete;

  DistributedPartitionedGraph partition() final;

private:
  const DistributedGraph &_graph;
  const Context &_ctx;
};
} // namespace kaminpar::dist
