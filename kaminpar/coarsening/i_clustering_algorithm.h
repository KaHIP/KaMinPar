/*******************************************************************************
 * @file:   i_clustering_algorithm.h
 *
 * @author: Daniel Seemaier
 * @date:   29.09.21
 * @brief:  Interface for clustering algorithms used for coarsening.
 ******************************************************************************/
#pragma once

#include "kaminpar/datastructure/graph.h"
#include "kaminpar/definitions.h"
#include "kaminpar/parallel.h"

namespace kaminpar {
class IClusteringAlgorithm {
public:
  using AtomicClusterArray = scalable_vector<parallel::IntegralAtomicWrapper<NodeID>>;

  IClusteringAlgorithm() = default;
  virtual ~IClusteringAlgorithm() = default;

  IClusteringAlgorithm(const IClusteringAlgorithm &) = delete;
  IClusteringAlgorithm &operator=(const IClusteringAlgorithm &) = delete;
  IClusteringAlgorithm(IClusteringAlgorithm &&) noexcept = default;
  IClusteringAlgorithm &operator=(IClusteringAlgorithm &&) noexcept = default;

  //
  // Optional options
  //

  virtual void set_max_cluster_weight(const NodeWeight /* weight */) {}
  virtual void set_desired_cluster_count(const NodeID /* count */) {}

  //
  // Clustering function
  //

  virtual const AtomicClusterArray &compute_clustering(const Graph &graph) = 0;
};
} // namespace kaminpar