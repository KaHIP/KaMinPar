/*******************************************************************************
 * Worker-local two-hop matching and clustering.
 *
 * @file:   threadwise_two_hop_clustering.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/algorithms/label_propagation/clustering_state.h"
#include "kaminpar-shm/algorithms/label_propagation/two_hop_candidates.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/dynamic_map.h"

namespace kaminpar::shm::lp {

/*!
 * Groups candidates through worker-local favored-cluster maps.
 *
 * This avoids cross-worker synchronization at the cost of allowing one group
 * per worker for the same favored cluster.
 */
template <typename Graph> class ThreadwiseTwoHopClustering {
public:
  ThreadwiseTwoHopClustering(const Graph &graph, ClusteringState &state)
      : _graph(graph),
        _state(state),
        _candidates(graph, state) {}

  void match() {
    run<true>();
  }

  void cluster() {
    run<false>();
  }

private:
  template <bool kMatching> void run() {
    tbb::enumerable_thread_specific<DynamicFlatMap<NodeID, NodeID>> groups;

    tbb::parallel_for(
        tbb::blocked_range<NodeID>(0, _graph.n(), 512),
        [&](const tbb::blocked_range<NodeID> &range) {
          auto &local_groups = groups.local();
          for (NodeID u = range.begin(); u != range.end(); ++u) {
            if (!_candidates.contains(u)) {
              continue;
            }

            const NodeID from = _state.cluster(u);
            NodeID &representative = local_groups[_candidates.group(u)];
            if (representative == 0) {
              representative = from + 1;
              continue;
            }

            const NodeID to = representative - 1;
            const bool moved = _state.move_cluster_weight(from, to, _state.cluster_weight(from));

            if constexpr (kMatching) {
              KASSERT(moved);
              _state.move_node(u, to);
              representative = 0;
            } else if (moved) {
              _state.move_node(u, to);
            } else {
              representative = from + 1;
            }
          }
        }
    );
  }

  const Graph &_graph;
  ClusteringState &_state;
  TwoHopCandidates<Graph> _candidates;
};

} // namespace kaminpar::shm::lp
