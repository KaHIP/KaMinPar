/*******************************************************************************
 * Deep multilevel graph partitioner that can use multiple v-cycles.
 *
 * @file:   vcycle_deep_multilevel.h
 * @author: Daniel Seemaier
 * @date:   31.10.2024
 ******************************************************************************/
#pragma once

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/partitioning/partitioner.h"

namespace kaminpar::shm {

class VcycleDeepMultilevelPartitioner : public Partitioner {
public:
  VcycleDeepMultilevelPartitioner(const Graph &input_graph, const Context &input_ctx);

  VcycleDeepMultilevelPartitioner(const VcycleDeepMultilevelPartitioner &) = delete;
  VcycleDeepMultilevelPartitioner &
  operator=(const VcycleDeepMultilevelPartitioner &) = delete;

  VcycleDeepMultilevelPartitioner(VcycleDeepMultilevelPartitioner &&) = delete;
  VcycleDeepMultilevelPartitioner &operator=(VcycleDeepMultilevelPartitioner &&) = delete;

  PartitionedGraph partition() final;

private:
  const Graph &_input_graph;
  const Context &_input_ctx;
};

} // namespace kaminpar::shm
