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

void HEMLPClusterer::set_max_cluster_weight(const GlobalNodeWeight weight) {
  _lp->set_max_cluster_weight(weight);
  _hem->set_max_cluster_weight(weight);
}

void HEMLPClusterer::cluster(StaticArray<GlobalNodeID> &clustering, const DistributedGraph &graph) {
  _graph = &graph;

  if (_fallback) {
    _lp->cluster(clustering, graph);
  } else {
    _hem->cluster(clustering, graph);

    // If the matching shrinks the graph by less than 10%, switch to label propagation
    // @todo make this configurable
    const GlobalNodeID new_size = compute_size_after_matching_contraction(clustering);
    if (1.0 * new_size / graph.global_n() > 0.9) {
      _fallback = true;
      cluster(clustering, graph);
    }
  }
}

GlobalNodeID
HEMLPClusterer::compute_size_after_matching_contraction(const StaticArray<GlobalNodeID> &clustering
) {
  tbb::enumerable_thread_specific<NodeID> num_matched_edges_ets;
  _graph->reified([&](const auto &graph) {
    graph.pfor_nodes([&](const NodeID u) {
      if (clustering[u] != graph.local_to_global_node(u)) {
        ++num_matched_edges_ets.local();
      }
    });
  });

  const NodeID num_matched_edges = num_matched_edges_ets.combine(std::plus{});
  const GlobalNodeID num_matched_edges_globally =
      mpi::allreduce<GlobalNodeID>(num_matched_edges, MPI_SUM, _graph->communicator());

  return _graph->global_n() - num_matched_edges_globally;
}

} // namespace kaminpar::dist
