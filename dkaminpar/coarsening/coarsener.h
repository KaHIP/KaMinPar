/*******************************************************************************
 * @file:   coarsener.h
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 * @brief:  Builds and manages a hierarchy of coarse graphs.
 ******************************************************************************/
#pragma once

#include <vector>

#include "dkaminpar/coarsening/clustering/clustering_algorithm.h"
#include "dkaminpar/coarsening/contraction/legacy_cluster_contraction.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist {
class Coarsener {
public:
    Coarsener(const DistributedGraph& input_graph, const Context& input_ctx);

    const DistributedGraph* coarsen_once();

    const DistributedGraph* coarsen_once(GlobalNodeWeight max_cluster_weight);

    DistributedPartitionedGraph uncoarsen_once(DistributedPartitionedGraph&& p_graph);

    GlobalNodeWeight        max_cluster_weight() const;
    const DistributedGraph* coarsest() const;
    std::size_t             level() const;

private:
    const DistributedGraph* coarsen_once_local(GlobalNodeWeight max_cluster_weight);
    const DistributedGraph* coarsen_once_global(GlobalNodeWeight max_cluster_weight);

    DistributedPartitionedGraph uncoarsen_once_local(DistributedPartitionedGraph&& p_graph);
    DistributedPartitionedGraph uncoarsen_once_global(DistributedPartitionedGraph&& p_graph);

    const DistributedGraph* nth_coarsest(std::size_t n) const;

    bool has_converged(const DistributedGraph& before, const DistributedGraph& after) const;

    const DistributedGraph& _input_graph;
    const Context&          _input_ctx;

    std::unique_ptr<ClusteringAlgorithm<GlobalNodeID>> _global_clustering_algorithm;
    std::unique_ptr<ClusteringAlgorithm<NodeID>>       _local_clustering_algorithm;

    std::vector<DistributedGraph>        _graph_hierarchy;
    std::vector<GlobalMapping>           _global_mapping_hierarchy; //< produced by global clustering algorithm
    std::vector<MigratedNodes>           _node_migration_history;
    std::vector<scalable_vector<NodeID>> _local_mapping_hierarchy; //< produced by local clustering_algorithm

    bool _local_clustering_converged = false;
};
} // namespace kaminpar::dist
