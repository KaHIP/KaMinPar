/*******************************************************************************
 * @file:   refiner.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Interface for refinement algorithms.
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_partitioned_graph.h"

namespace kaminpar::dist {
class GlobalRefiner {
public:
  virtual ~GlobalRefiner() = default;

  virtual void initialize() = 0;
  virtual bool refine() = 0;
};

template <typename Refiner, typename Factory> class ExclusiveGlobalRefiner : public GlobalRefiner {
public:
  ExclusiveGlobalRefiner(Factory &factory) : _factory(factory) {}

  virtual ~ExclusiveGlobalRefiner() {
    _factory.reclaim(*static_cast<Refiner *>(this));
  }

private:
  Factory &_factory;
};

class GlobalRefinerFactory {
public:
  virtual ~GlobalRefinerFactory() = default;

  virtual std::unique_ptr<GlobalRefiner>
  create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) = 0;
};
} // namespace kaminpar::dist
