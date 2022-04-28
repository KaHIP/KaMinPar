/*******************************************************************************
 * @file:   coarsener.h
 *
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 * @brief:  Builds and manages a hierarchy of coarse graphs.
 ******************************************************************************/
#pragma once

#include <vector>

#include "dkaminpar/coarsening/global_clustering_contraction.h"
#include "dkaminpar/coarsening/i_clustering.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructure/distributed_graph.h"

namespace dkaminpar {
class Coarsener {
public:
    Coarsener(const DistributedGraph& input_graph, const Context& input_ctx);

    const DistributedGraph* coarsen_once();

    const DistributedGraph* coarsen_once(const GlobalNodeWeight max_cluster_weight);

    DistributedPartitionedGraph uncoarsen_once(DistributedPartitionedGraph&& p_graph);

    GlobalNodeWeight        max_cluster_weight() const;
    const DistributedGraph* coarsest() const;
    std::size_t             level() const;

private:
    const DistributedGraph* nth_coarsest(std::size_t n) const;

    const DistributedGraph& _input_graph;
    const Context&          _input_ctx;

    std::unique_ptr<ClusteringAlgorithm> _clustering_algorithm;

    std::vector<DistributedGraph>          _graph_hierarchy;
    std::vector<coarsening::GlobalMapping> _mapping_hierarchy;
};
} // namespace dkaminpar
