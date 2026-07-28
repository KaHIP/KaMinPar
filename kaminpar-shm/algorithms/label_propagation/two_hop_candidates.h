/*******************************************************************************
 * Candidate selection shared by two-hop clustering algorithms.
 *
 * @file:   two_hop_candidates.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/algorithms/label_propagation/clustering_state.h"

namespace kaminpar::shm::lp {

/*!
 * Identifies unchanged, light singleton clusters that may be combined through
 * a common favored cluster.
 */
template <typename Graph> class TwoHopCandidates {
public:
  TwoHopCandidates(const Graph &graph, const ClusteringState &state)
      : _graph(graph),
        _state(state) {}

  [[nodiscard]] bool contains(const NodeID u) const {
    if (_graph.degree(u) == 0 || _state.cluster(u) != u) {
      return false;
    }

    const NodeWeight weight = _state.cluster_weight(u);
    return weight <= _state.max_cluster_weight(u) / 2 && weight == _graph.node_weight(u);
  }

  [[nodiscard]] NodeID group(const NodeID u) const {
    return _state.favored_cluster(u);
  }

private:
  const Graph &_graph;
  const ClusteringState &_state;
};

} // namespace kaminpar::shm::lp
