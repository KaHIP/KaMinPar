/*******************************************************************************
 * Interface for refinement algorithms.
 *
 * @file:   refiner.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"

namespace kaminpar::dist {
/**
 * Interface for refinement algorithms. The graph partition and partition
 * context struct should be passed via the ctor.
 *
 * Refiners should be instantiated via the corresponding factory class.
 */
class GlobalRefiner {
public:
  virtual ~GlobalRefiner() = default;

  /**
   * Initializes data structures that are static for the graph.
   * This function is called only once before the first call to refine().
   */
  virtual void initialize() = 0;

  /**
   * Refines the graph partition. The return value indicates whether the refiner
   * improved the partition. This function might be called multiple times.
   *
   * @return true if the partition was improved, false otherwise.
   */
  virtual bool refine() = 0;
};

/**
 * Factory interface for refinement algorithms. Factories create refiner instances
 * for a given graph partition and partition context.
 *
 * The use of factories allow resources to be re-used.
 */
class GlobalRefinerFactory {
public:
  virtual ~GlobalRefinerFactory() = default;

  virtual std::unique_ptr<GlobalRefiner>
  create(DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx) = 0;
};
} // namespace kaminpar::dist
