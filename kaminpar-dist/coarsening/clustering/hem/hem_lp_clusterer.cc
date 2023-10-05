/*******************************************************************************
 * Clustering via heavy edge matching with label propagation fallback.
 *
 * @file:   hem_lp_clusterer.cc
 * @author: Daniel Seemaier
 * @date:   19.12.2022
 ******************************************************************************/
#include "kaminpar-dist/coarsening/clustering/hem/hem_lp_clusterer.h"

#include <functional>

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/coarsening/clustering/hem/hem_clusterer.h"
#include "kaminpar-dist/coarsening/clustering/lp/global_lp_clusterer.h"

namespace kaminpar::dist {
HEMLPClusterer::HEMLPClusterer(const Context &ctx)
    : _lp(std::make_unique<GlobalLPClusterer>(ctx)),
      _hem(std::make_unique<HEMClusterer>(ctx)) {}

void HEMLPClusterer::initialize(const DistributedGraph &graph) {
  _lp->initialize(graph);
  _hem->initialize(graph);
}

HEMLPClusterer::ClusterArray &
HEMLPClusterer::cluster(const DistributedGraph &graph, const GlobalNodeWeight max_cluster_weight) {
  _graph = &graph;

  if (_fallback) {
    return _lp->cluster(graph, max_cluster_weight);
  } else {
    auto &matching = _hem->cluster(graph, max_cluster_weight);
    const GlobalNodeID new_size = compute_size_after_matching_contraction(matching);

    // @todo make this configurable
    if (1.0 * new_size / graph.global_n() <= 0.9) { // Shrink by at least 10%
      return matching;
    }

    _fallback = true;
    return cluster(graph, max_cluster_weight);
  }
}

GlobalNodeID HEMLPClusterer::compute_size_after_matching_contraction(const ClusterArray &clustering
) {
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
