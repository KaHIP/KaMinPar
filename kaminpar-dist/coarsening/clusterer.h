/*******************************************************************************
 * Interface for clustering algorithms.
 *
 * @file:   clusterer.h
 * @author: Daniel Seemaier
 * @date:   29.09.21
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::dist {

class Clusterer {
public:
  Clusterer() = default;

  Clusterer(const Clusterer &) = delete;
  Clusterer &operator=(const Clusterer &) = delete;

  Clusterer(Clusterer &&) noexcept = default;
  Clusterer &operator=(Clusterer &&) noexcept = default;

  virtual ~Clusterer() = default;

  //
  // Optional options
  //

  virtual void set_communities(const StaticArray<BlockID> & /* communities */) {}
  virtual void clear_communities() {}

  virtual void set_max_cluster_weight(GlobalNodeWeight /* weight */) {}

  //
  // Clustering function
  //

  virtual void cluster(StaticArray<GlobalNodeID> &clustering, const DistributedGraph &graph) = 0;
};

} // namespace kaminpar::dist
