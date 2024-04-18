/*******************************************************************************
 * Interface for the coarsening phase of multilevel graph partitioning.
 *
 * @file:   coarsener.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {
/**
 * Interface for the coarsening phase of multilevel graph partitioning.
 */
class Coarsener {
public:
  Coarsener() = default;

  Coarsener(const Coarsener &) = delete;
  Coarsener &operator=(const Coarsener &) = delete;

  Coarsener(Coarsener &&) noexcept = default;
  Coarsener &operator=(Coarsener &&) noexcept = default;

  virtual ~Coarsener() = default;

  /**
   * Initializes the coarsener with a new toplevel graph.
   */
  virtual void initialize(const Graph *graph) = 0;

  /**
   * Computes the next level of the graph hierarchy.
   *
   * @return whether coarsening has *not* yet converged.
   */
  virtual bool coarsen() = 0;

  /**
   * @return the coarsest graph in the hierarchy.
   */
  [[nodiscard]] virtual const Graph &current() const = 0;

  /**
   * @return number of coarse graphs in the hierarchy.
   */
  [[nodiscard]] virtual std::size_t level() const = 0;

  /**
   * @return whether we have *not* yet computed any coarse graphs.
   */
  [[nodiscard]] bool empty() const {
    return level() == 0;
  }

  /**
   * Projects a partition of the currently coarsest graph onto the next finer
   * graph and frees the currently coarsest graph, i.e., unrolls one level of
   * the coarse graph hierarchy.
   *
   * @param p_graph Partition of the currently coarsest graph.
   *                Precondition: `p_graph.graph() == current()`.
   *
   * @return partition of the *new* coarsest graph.
   */
  virtual PartitionedGraph uncoarsen(PartitionedGraph &&p_graph) = 0;
};
} // namespace kaminpar::shm
