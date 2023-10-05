/*******************************************************************************
 * Interface for graph coarseners.
 *
 * @file:   coarsener.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/coarsening/clusterer.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/definitions.h"

namespace kaminpar::shm {
/**
 * Clustering graphutils.
 *
 * Call #coarsen() repeatedly to produce a hierarchy of coarse graph. The coarse
 * graphs are owned by the clustering graphutils. To unroll the graph hierarchy,
 * call #uncoarsen() with a partition of the currently coarsest graph.
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
   * Coarsen the currently coarsest graph with a static maximum node weight.
   *
   * @param max_cluster_weight Maximum node weight of the coarse graph.
   * @param to_size Desired size of the coarse graph.
   * @return New coarsest graph and whether coarsening has not converged.
   */
  virtual std::pair<const Graph *, bool>
  compute_coarse_graph(NodeWeight max_cluster_weight, NodeID to_size) = 0;

  /** @return The currently coarsest graph, or the input graph, if no coarse
   * graphs have been computed so far. */
  [[nodiscard]] virtual const Graph *coarsest_graph() const = 0;

  /** @return Number of coarsest graphs that have already been computed. */
  [[nodiscard]] virtual std::size_t size() const = 0;

  /** @return Whether we have not computed any coarse graphs so far. */
  [[nodiscard]] bool empty() const {
    return size() == 0;
  }

  /**
   * Projects a partition of the currently coarsest graph onto the next finer
   * graph and frees the currently coarsest graph, i.e., unrolls one level of
   * the coarse graph hierarchy.
   *
   * @param p_graph Partition of the currently coarsest graph, i.e.,
   * `p_graph.graph() == *coarsest_graph()`.
   * @return Partition of the new coarsest graph.
   */
  virtual PartitionedGraph uncoarsen(PartitionedGraph &&p_graph) = 0;

  //! Re-initialize this coarsener object with a new graph.
  virtual void initialize(const Graph *graph) = 0;
};
} // namespace kaminpar::shm
