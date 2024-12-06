/*******************************************************************************
 * Common code for graph contraction algorithms.
 *
 * @file:   contraction.h
 * @author: Daniel Seemaier
 * @date:   06.05.2024
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::dist {

class CoarseGraph {
public:
  virtual ~CoarseGraph() = default;

  [[nodiscard]] virtual const DistributedGraph &get() const = 0;
  [[nodiscard]] virtual DistributedGraph &get() = 0;

  virtual void project(const StaticArray<BlockID> &partition, StaticArray<BlockID> &onto) = 0;
};

} // namespace kaminpar::dist
