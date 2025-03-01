/*******************************************************************************
 * Multilevel graph partitioning with direct k-way initial partitioning.
 *
 * @file:   kway_partitioner.h
 * @author: Daniel Seemaier
 * @date:   25.10.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/partitioning/partitioner.h"

namespace kaminpar::dist {

class KWayMultilevelPartitioner : public Partitioner {
public:
  KWayMultilevelPartitioner(const DistributedGraph &input_graph, const Context &input_ctx);

  KWayMultilevelPartitioner(const KWayMultilevelPartitioner &) = delete;
  KWayMultilevelPartitioner &operator=(const KWayMultilevelPartitioner &) = delete;

  KWayMultilevelPartitioner(KWayMultilevelPartitioner &&) noexcept = default;
  KWayMultilevelPartitioner &operator=(KWayMultilevelPartitioner &&) = delete;

  DistributedPartitionedGraph partition() final;

private:
  const DistributedGraph &_input_graph;
  const Context &_input_ctx;
};

} // namespace kaminpar::dist
