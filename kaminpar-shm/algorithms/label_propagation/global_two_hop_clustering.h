/*******************************************************************************
 * Globally coordinated two-hop matching and clustering.
 *
 * @file:   global_two_hop_clustering.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <tbb/parallel_for.h>

#include "kaminpar-shm/algorithms/label_propagation/clustering_state.h"
#include "kaminpar-shm/algorithms/label_propagation/two_hop_candidates.h"

#include "kaminpar-common/assert.h"

namespace kaminpar::shm::lp {

/*!
 * Groups candidates that share a favored cluster across all workers.
 *
 * Matching closes a group after two candidates. Clustering keeps appending to
 * the current group until its weight limit is reached.
 */
template <typename Graph> class GlobalTwoHopClustering {
public:
  GlobalTwoHopClustering(const Graph &graph, ClusteringState &state)
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
    prepare_groups();
    check_precondition();

    tbb::parallel_for<NodeID>(0, _graph.n(), [&](const NodeID u) {
      if (_state.should_stop() || !_candidates.contains(u)) {
        return;
      }

      const NodeID group = _candidates.group(u);
      NodeID current = _state.favored_cluster(group);

      do {
        if (current == group) {
          if (_state.compare_exchange_favored_cluster(group, current, u)) {
            break;
          }
          if (current == group) {
            continue;
          }
        }

        KASSERT(_state.favored_cluster(current) == group);

        if constexpr (kMatching) {
          if (_state.compare_exchange_favored_cluster(group, current, group)) {
            [[maybe_unused]] const bool moved =
                _state.move_cluster_weight(u, current, _state.cluster_weight(u));
            KASSERT(moved);
            _state.move_node(u, current);
            _state.remove_empty_clusters(1);
            break;
          }
        } else {
          if (_state.move_cluster_weight(u, current, _state.cluster_weight(u))) {
            _state.move_node(u, current);
            _state.remove_empty_clusters(1);
            break;
          }

          if (_state.compare_exchange_favored_cluster(group, current, group)) {
            break;
          }
        }
      } while (true);
    });
  }

  void prepare_groups() {
    tbb::parallel_for<NodeID>(0, _graph.n(), [&](const NodeID u) {
      if (_candidates.contains(u)) {
        const NodeID favored = _candidates.group(u);
        if (favored != u && _candidates.contains(favored) &&
            _state.move_cluster_weight(u, favored, _state.cluster_weight(u))) {
          _state.move_node(u, favored);
          _state.remove_empty_clusters(1);
        }
      } else {
        _state.set_favored_cluster(u, u);
      }
    });
  }

  void check_precondition() const {
    KASSERT(
        [&] {
          for (NodeID u = 0; u < _graph.n(); ++u) {
            const NodeID favored = _candidates.group(u);
            if (u != favored && _candidates.contains(u) && _candidates.contains(favored)) {
              return false;
            }
          }
          return true;
        }(),
        "precondition for two-hop clustering violated",
        assert::heavy
    );
  }

  const Graph &_graph;
  ClusteringState &_state;
  TwoHopCandidates<Graph> _candidates;
};

} // namespace kaminpar::shm::lp
