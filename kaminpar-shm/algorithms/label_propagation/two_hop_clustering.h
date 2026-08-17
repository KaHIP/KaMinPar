/*******************************************************************************
 * Reusable two-hop clustering algorithms.
 *
 * @file:   two_hop_clustering.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/algorithms/label_propagation/clustering_state.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/dynamic_map.h"

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
    struct LocalState {
      DynamicFlatMap<NodeID, NodeID> groups;
      NodeID removed_clusters = 0;
    };
    tbb::enumerable_thread_specific<LocalState> local_states;

    tbb::parallel_for(
        tbb::blocked_range<NodeID>(0, _graph.n(), 512),
        [&](const tbb::blocked_range<NodeID> &range) {
          auto &local = local_states.local();
          for (NodeID u = range.begin(); u != range.end(); ++u) {
            if (!_candidates.contains(u)) {
              continue;
            }

            const NodeID from = _state.cluster(u);
            NodeID &representative = local.groups[_candidates.group(u)];
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
              ++local.removed_clusters;
            } else if (moved) {
              _state.move_node(u, to);
              ++local.removed_clusters;
            } else {
              representative = from + 1;
            }
          }
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
  TwoHopCandidates<Graph> _candidates;
};

} // namespace kaminpar::shm::lp
