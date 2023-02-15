/*******************************************************************************
 * @file:   hem_lp_clustering.cc
 * @author: Daniel Seemaier
 * @date:   19.12.2022
 * @brief:  Clustering using a combination of label propagation and heavy edge
 * matching.
 ******************************************************************************/
#include "dkaminpar/coarsening/clustering/hem_lp_clustering.h"

#include <functional>

#include <tbb/enumerable_thread_specific.h>

#include "dkaminpar/coarsening/clustering/global_label_propagation_clustering.h"
#include "dkaminpar/coarsening/clustering/hem_clustering.h"
#include "dkaminpar/mpi/wrapper.h"

namespace kaminpar::dist {
HEMLPClustering::HEMLPClustering(const Context& ctx)
    : _lp(std::make_unique<DistributedGlobalLabelPropagationClustering>(ctx)),
      _hem(std::make_unique<HEMClustering>(ctx)) {}

const HEMLPClustering::AtomicClusterArray&
HEMLPClustering::compute_clustering(const DistributedGraph& graph, const GlobalNodeWeight max_cluster_weight) {
    _graph = &graph;

    if (_fallback) {
        return _lp->compute_clustering(graph, max_cluster_weight);
    } else {
        const auto&        matching = _hem->compute_clustering(graph, max_cluster_weight);
        const GlobalNodeID new_size = compute_size_after_matching_contraction(matching);

        // @todo make this configurable
        if (1.0 * new_size / graph.global_n() <= 0.9) { // Shrink by at least 10%
            return matching;
        }

        _fallback = true;
        return compute_clustering(graph, max_cluster_weight);
    }
}

GlobalNodeID HEMLPClustering::compute_size_after_matching_contraction(const AtomicClusterArray& clustering) {
    tbb::enumerable_thread_specific<NodeID> num_matched_edges_ets;
    _graph->pfor_nodes([&](const NodeID u) {
        if (clustering[u] != _graph->local_to_global_node(u)) {
            ++num_matched_edges_ets.local();
        }
    });
    const NodeID num_matched_edges = num_matched_edges_ets.combine(std::plus{});

    const GlobalNodeID num_matched_edges_globally =
        mpi::allreduce<GlobalNodeID>(num_matched_edges, MPI_SUM, _graph->communicator());

    return _graph->global_n() - num_matched_edges_globally;
}
} // namespace kaminpar::dist
