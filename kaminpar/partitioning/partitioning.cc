/*******************************************************************************
 * @file:   partitioning.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:
 ******************************************************************************/
#include "kaminpar/partitioning/partitioning.h"

#include "kaminpar/partitioning/deep_multilevel.h"
#include "kaminpar/partitioning/rb_multilevel.h"

#include "common/logger.h"
#include "common/timer.h"

namespace kaminpar::shm::partitioning {
PartitionedGraph partition(const Graph &graph, const Context &ctx) {
  switch (ctx.mode) {
  case PartitioningMode::DEEP: {
    DeepMultilevelPartitioner deep(graph, ctx);
    PartitionedGraph p_graph = deep.partition();
    return p_graph;
  }

  case PartitioningMode::RB: {
    RBMultilevelPartitioner rb(graph, ctx);
    PartitionedGraph p_graph = rb.partition();
    return p_graph;
  }

  default:
    FATAL_ERROR << "Unsupported mode";
  }
  __builtin_unreachable();
}
} // namespace kaminpar::shm::partitioning
