/*******************************************************************************
 * Isolated-node and two-hop clustering postprocessing.
 *
 * @file:   clustering_postprocessor.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <limits>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/algorithms/label_propagation/clustering_state.h"
#include "kaminpar-shm/algorithms/label_propagation/two_hop_clustering.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm::lp {

template <typename Graph> class ClusteringPostprocessor {
public:
  ClusteringPostprocessor(
      const Graph &graph,
      ClusteringState &state,
      const TwoHopStrategy two_hop_strategy,
      const double two_hop_threshold
  )
      : _graph(graph),
        _state(state),
        _two_hop_strategy(two_hop_strategy),
        _two_hop_threshold(two_hop_threshold) {}

  void run() {
    if (!should_run()) {
      return;
    }

    match_isolated_nodes();
    switch (_two_hop_strategy) {
    case TwoHopStrategy::MATCH:
      GlobalTwoHopClustering(_graph, _state).match();
      break;
    case TwoHopStrategy::MATCH_THREADWISE:
      ThreadwiseTwoHopClustering(_graph, _state).match();
      break;
    case TwoHopStrategy::CLUSTER:
      GlobalTwoHopClustering(_graph, _state).cluster();
      break;
    case TwoHopStrategy::CLUSTER_THREADWISE:
      ThreadwiseTwoHopClustering(_graph, _state).cluster();
      break;
    }
  }

private:
  [[nodiscard]] bool should_run() const {
    return (1.0 - 1.0 * _state.num_clusters() / _graph.n()) <= _two_hop_threshold;
  }

  void match_isolated_nodes() {
    constexpr NodeID kInvalidCluster = std::numeric_limits<NodeID>::max();
    struct LocalState {
      NodeID current_cluster = std::numeric_limits<NodeID>::max();
      NodeID removed_clusters = 0;
    };
    tbb::enumerable_thread_specific<LocalState> local_states;

    tbb::parallel_for(
        tbb::blocked_range<NodeID>(0, _graph.n()), [&](const tbb::blocked_range<NodeID> &range) {
          auto &local = local_states.local();
          NodeID cluster = local.current_cluster;
          for (NodeID u = range.begin(); u != range.end(); ++u) {
            if (_graph.degree(u) != 0) {
              continue;
            }

            const NodeID from = _state.cluster(u);
            if (cluster != kInvalidCluster &&
                _state.move_cluster_weight(from, cluster, _state.cluster_weight(from))) {
              _state.move_node(u, cluster);
              cluster = kInvalidCluster;
              ++local.removed_clusters;
            } else {
              cluster = from;
            }
          }
          local.current_cluster = cluster;
        }
    );

    NodeID removed_clusters = 0;
    for (const auto &local : local_states) {
      removed_clusters += local.removed_clusters;
    }
    _state.remove_empty_clusters(removed_clusters);
  }

  const Graph &_graph;
  ClusteringState &_state;
  TwoHopStrategy _two_hop_strategy;
  double _two_hop_threshold;
};

} // namespace kaminpar::shm::lp
