/*******************************************************************************
 * Builds and manages a hierarchy of coarse graphs.
 *
 * @file:   coarsener.h
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 ******************************************************************************/
#pragma once

#include <vector>

#include "kaminpar-dist/coarsening/clustering/clusterer.h"
#include "kaminpar-dist/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-dist/context.h"
#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_partitioned_graph.h"
#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-common/datastructures/scalable_vector.h"

namespace kaminpar::dist {
class Coarsener {
public:
  Coarsener(const DistributedGraph &input_graph, const Context &input_ctx);

  const DistributedGraph *coarsen_once();

  const DistributedGraph *coarsen_once(GlobalNodeWeight max_cluster_weight);

  DistributedPartitionedGraph uncoarsen_once(DistributedPartitionedGraph &&p_graph);

  GlobalNodeWeight max_cluster_weight() const;
  const DistributedGraph *coarsest() const;
  std::size_t level() const;

private:
  const DistributedGraph *coarsen_once_local(GlobalNodeWeight max_cluster_weight);
  const DistributedGraph *coarsen_once_global(GlobalNodeWeight max_cluster_weight);

  DistributedPartitionedGraph uncoarsen_once_local(DistributedPartitionedGraph &&p_graph);
  DistributedPartitionedGraph uncoarsen_once_global(DistributedPartitionedGraph &&p_graph);

  const DistributedGraph *nth_coarsest(std::size_t n) const;

  bool has_converged(const DistributedGraph &before, const DistributedGraph &after) const;

  const DistributedGraph &_input_graph;
  const Context &_input_ctx;

  std::unique_ptr<GlobalClusterer> _global_clusterer;
  std::unique_ptr<LocalClusterer> _local_clusterer;

  std::vector<DistributedGraph> _graph_hierarchy;
  std::vector<GlobalMapping> _global_mapping_hierarchy; //< produced by global clustering algorithm
  std::vector<MigratedNodes> _node_migration_history;
  std::vector<scalable_vector<NodeID>>
      _local_mapping_hierarchy; //< produced by local clustering_algorithm

  bool _local_clustering_converged = false;
};
} // namespace kaminpar::dist
