#pragma once

#include "datastructure/graph.h"
#include "definitions.h"

#include <functional>

namespace kaminpar {
/**
 * Clustering algorithm.
 *
 * Call #coarsen() repeatedly to produce a hierarchy of coarse graph. The coarse graphs are owned by the clustering
 * algorithm. To unroll the graph hierarchy, call #uncoarsen() with a partition of the currently coarsest graph.
 */
class Coarsener {
public:
  Coarsener() = default;
  virtual ~Coarsener() = default;

  Coarsener(Coarsener &&) noexcept = default;
  Coarsener &operator=(Coarsener &&) noexcept = default;

  /**
   * Coarsen the currently coarsest graph with a static maximum node weight.
   *
   * @param max_cluster_weight Maximum node weight of the coarse graph.
   * @return New coarsest graph and whether coarsening has not converged.
   */
  virtual std::pair<const Graph *, bool> coarsen(const NodeWeight max_cluster_weight) {
    return coarsen([max_cluster_weight](NodeID) { return max_cluster_weight; });
  };

  /**
   * Coarsen the currently coarsest graph with a flexible maximum node weight callback. This is useful when interleaving
   * coarse graph construction with clustering: in this case, the callback is invoked twice, once when computing a
   * clustering of the currently coarsest graph and once when computing a clustering of the computed coarsest graph.
   *
   * @param cb_max_cluster_weight Maximum node weight callback: takes the number of nodes and returns the maximum node
   * weight to be used.
   * @return New coarsest graph and whether coarsening has not converged.
   */
  virtual std::pair<const Graph *, bool> coarsen(const std::function<NodeWeight(NodeID)> &cb_max_cluster_weight) = 0;

  /** @return The currently coarsest graph, or the input graph, if no coarse graphs have been computed so far. */
  virtual const Graph *coarsest_graph() const = 0;

  /** @return Number of coarsest graphs that have already been computed. */
  virtual std::size_t size() const = 0;

  /** @return Whether we have not computed any coarse graphs so far. */
  [[nodiscard]] bool empty() const { return size() == 0; }

  /**
   * Projects a partition of the currently coarsest graph onto the next finer graph and frees the currently coarsest
   * graph, i.e., unrolls one level of the coarse graph hierarchy.
   *
   * @param p_graph Partition of the currently coarsest graph, i.e., `p_graph.graph() == *coarsest_graph()`.
   * @return Partition of the new coarsest graph.
   */
  virtual PartitionedGraph uncoarsen(PartitionedGraph &&p_graph) = 0;

  /**
   * A community structure restricts coarsening algorithms: if two nodes u, v belong to different communities, i.e.,
   * if `communities[u] != communities[v]`, the nodes may not be merged into the same coarse node. This can be used to
   * implemented V-cycles: using a pre-existing partition of the graph as community structure only allows nodes within
   * the same block to be contracted, and therefore, a partition of the finest graph can be projected onto the coarsest
   * graph.
   *
   * During coarsening, the community structure is also coarsened. Uncoarsening implicitly discards the community
   * structure. If no community is set, it is ignored.
   *
   * @param communities If set, the coarsening algorithm cannot contract nodes that belong to different communities.
   */
  virtual void set_community_structure(std::vector<BlockID> communities) = 0;

  //! Re-initialize this coarsener object with a new graph.
  virtual void initialize(const Graph *graph) = 0;
};

/**
 * Dummy coarsening that converges immediately.
 */
class NoopCoarsener : public Coarsener {
public:
  using Coarsener::coarsen;

  void initialize(const Graph *graph) final { _graph = graph; }

  std::pair<const Graph *, bool> coarsen(const std::function<NodeWeight(NodeID)> &) final {
    return {coarsest_graph(), false};
  }

  std::size_t size() const final { return 0; }

  const Graph *coarsest_graph() const final { return _graph; }

  PartitionedGraph uncoarsen(PartitionedGraph &&p_graph) final { return std::move(p_graph); }

  void set_community_structure(std::vector<BlockID>) final {}

private:
  const Graph *_graph{nullptr};
};
} // namespace kaminpar