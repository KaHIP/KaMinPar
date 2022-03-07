/*******************************************************************************
 * @file:   partitioning.cc
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Interface for partitioning schemes.
 ******************************************************************************/
#include "dkaminpar/partitioning_scheme/partitioning.h"

#include "dkaminpar/partitioning_scheme/kway.h"

namespace dkaminpar {
DistributedPartitionedGraph partition(const DistributedGraph &graph, const Context &ctx) {
  switch (ctx.partition.mode) {
  case PartitioningMode::KWAY:
    return KWayPartitioningScheme{graph, ctx}.partition();
  case PartitioningMode::DEEP:
    FATAL_ERROR << "not implemented";
    break;
  case PartitioningMode::RB:
    FATAL_ERROR << "not implemented";
    break;
  }

  __builtin_unreachable();
}
} // namespace dkaminpar