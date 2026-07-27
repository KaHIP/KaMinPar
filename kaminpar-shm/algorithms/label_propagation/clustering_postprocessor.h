/*******************************************************************************
 * Preset-used isolated-node and two-hop clustering postprocessing.
 *
 * @file:   clustering_postprocessor.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <limits>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/algorithms/label_propagation/clustering_state.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/dynamic_map.h"

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
    if (_two_hop_strategy == TwoHopStrategy::MATCH_THREADWISE) {
      match_two_hop_nodes_threadwise();
    } else {
      KASSERT(_two_hop_strategy == TwoHopStrategy::CLUSTER);
      cluster_two_hop_nodes();
    }
  }

private:
  [[nodiscard]] bool should_run() const {
    return (1.0 - 1.0 * _state.num_clusters() / _graph.n()) <= _two_hop_threshold;
  }

  void match_isolated_nodes() {
    constexpr NodeID kInvalidCluster = std::numeric_limits<NodeID>::max();
    tbb::enumerable_thread_specific<NodeID> current_cluster_ets(kInvalidCluster);

    tbb::parallel_for(
        tbb::blocked_range<NodeID>(0, _graph.n()), [&](const tbb::blocked_range<NodeID> &range) {
          NodeID cluster = current_cluster_ets.local();
          for (NodeID u = range.begin(); u != range.end(); ++u) {
            if (_graph.degree(u) != 0) {
              continue;
            }

            const NodeID from = _state.cluster(u);
            if (cluster != kInvalidCluster &&
                _state.move_cluster_weight(from, cluster, _state.cluster_weight(from))) {
              _state.move_node(u, cluster);
              cluster = kInvalidCluster;
            } else {
              cluster = from;
            }
          }
          current_cluster_ets.local() = cluster;
        }
    );
  }

  [[nodiscard]] bool is_two_hop_candidate(const NodeID u) const {
    if (_graph.degree(u) == 0 || u != _state.cluster(u)) {
      return false;
    }

    const NodeWeight weight = _state.cluster_weight(u);
    return weight <= _state.max_cluster_weight(u) / 2 && weight == _graph.node_weight(u);
  }

  void match_two_hop_nodes_threadwise() {
    tbb::enumerable_thread_specific<DynamicFlatMap<NodeID, NodeID>> matching_maps;

    tbb::parallel_for(
        tbb::blocked_range<NodeID>(0, _graph.n(), 512),
        [&](const tbb::blocked_range<NodeID> &range) {
          auto &matching_map = matching_maps.local();
          for (NodeID u = range.begin(); u != range.end(); ++u) {
            if (!is_two_hop_candidate(u)) {
              continue;
            }

            const NodeID from = _state.cluster(u);
            NodeID &representative = matching_map[_state.favored_cluster(u)];
            if (representative == 0) {
              representative = from + 1;
              continue;
            }

            const NodeID to = representative - 1;
            const bool moved = _state.move_cluster_weight(from, to, _state.cluster_weight(from));
            KASSERT(moved);
            _state.move_node(u, to);
            representative = 0;
          }
        }
    );
  }

  void cluster_two_hop_nodes() {
    tbb::parallel_for<NodeID>(0, _graph.n(), [&](const NodeID u) {
      if (is_two_hop_candidate(u)) {
        const NodeID to = _state.favored_cluster(u);
        if (is_two_hop_candidate(to) &&
            _state.move_cluster_weight(u, to, _state.cluster_weight(u))) {
          _state.move_node(u, to);
          _state.remove_empty_clusters(1);
        }
      } else {
        _state.set_favored_cluster(u, u);
      }
    });

    KASSERT(
        [&] {
          for (NodeID u = 0; u < _graph.n(); ++u) {
            const NodeID favored = _state.favored_cluster(u);
            if (u != favored && is_two_hop_candidate(u) && is_two_hop_candidate(favored)) {
              return false;
            }
          }
          return true;
        }(),
        "precondition for two-hop clustering violated",
        assert::heavy
    );

    tbb::parallel_for<NodeID>(0, _graph.n(), [&](const NodeID u) {
      if (_state.should_stop() || !is_two_hop_candidate(u)) {
        return;
      }

      const NodeID favored = __atomic_load_n(&_state.favored_cluster_ref(u), __ATOMIC_RELAXED);
      NodeID &sync = _state.favored_cluster_ref(favored);

      do {
        NodeID cluster = sync;
        if (cluster == favored) {
          if (__atomic_compare_exchange_n(
                  &sync, &cluster, u, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
              )) {
            break;
          }
          if (cluster == favored) {
            continue;
          }
        }

        KASSERT(__atomic_load_n(&_state.favored_cluster_ref(cluster), __ATOMIC_RELAXED) == favored);

        if (_state.move_cluster_weight(u, cluster, _state.cluster_weight(u))) {
          _state.move_node(u, cluster);
          _state.remove_empty_clusters(1);
          break;
        }

        if (__atomic_compare_exchange_n(
                &sync, &cluster, favored, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
            )) {
          break;
        }
      } while (true);
    });
  }

  const Graph &_graph;
  ClusteringState &_state;
  TwoHopStrategy _two_hop_strategy;
  double _two_hop_threshold;
};

} // namespace kaminpar::shm::lp
