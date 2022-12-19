/*******************************************************************************
 * @file:   hem_clustering.h
 * @author: Daniel Seemaier
 * @date:   19.12.2022
 * @brief:  Clustering using heavy edge matching.
 ******************************************************************************/
#pragma once

#include "dkaminpar/algorithms/greedy_node_coloring.h"
#include "dkaminpar/coarsening/clustering_algorithm.h"
#include "dkaminpar/context.h"
#include "dkaminpar/definitions.h"

namespace kaminpar::dist {
class HEMClustering : public ClusteringAlgorithm<GlobalNodeID> {
public:
    HEMClustering(const Context& ctx);

    HEMClustering(const HEMClustering&)            = delete;
    HEMClustering& operator=(const HEMClustering&) = delete;
    HEMClustering(HEMClustering&&) noexcept        = default;
    HEMClustering& operator=(HEMClustering&&)      = delete;

    const AtomicClusterArray&
    compute_clustering(const DistributedGraph& graph, GlobalNodeWeight max_cluster_weight) final;

private:
    void initialize(const DistributedGraph& graph);

    void compute_local_matching(ColorID c, GlobalNodeWeight max_cluster_weight);
    void resolve_global_conflicts(ColorID c);
    void turn_into_clustering(ColorID c);

    const Context&              _input_ctx;
    const HEMCoarseningContext& _ctx;

    const DistributedGraph* _graph;

    AtomicClusterArray _matching;

    NoinitVector<std::uint8_t> _color_blacklist;
    NoinitVector<ColorID>      _color_sizes;
    NoinitVector<NodeID>       _color_sorted_nodes;

    std::vector<parallel::Atomic<std::uint8_t>> _matched;
};
} // namespace kaminpar::dist
