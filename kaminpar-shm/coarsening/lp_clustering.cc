/******************************************************************************
 * Label propagation for graph coarsening / clustering.
 *
 * @file:   lp_clustering.cc
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#include "kaminpar-shm/coarsening/lp_clustering.h"

#include <memory>

#include "kaminpar-shm/coarsening/clusterer.h"
#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/label_propagation.h"

#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
//
// Actual implementation -- not exposed in header
//

struct LPClusteringConfig : public LabelPropagationConfig {
  using ClusterID = NodeID;
  using ClusterWeight = BlockWeight;
  static constexpr bool kTrackClusterCount = true;
  static constexpr bool kUseTwoHopClustering = true;
};

class LPClusteringImpl final
    : public ChunkRandomdLabelPropagation<LPClusteringImpl, LPClusteringConfig>,
      public OwnedRelaxedClusterWeightVector<NodeID, NodeWeight>,
      public OwnedClusterVector<NodeID, NodeID>,
      public Clusterer {
  SET_DEBUG(false);

  using Base = ChunkRandomdLabelPropagation<LPClusteringImpl, LPClusteringConfig>;
  using ClusterWeightBase = OwnedRelaxedClusterWeightVector<NodeID, NodeWeight>;
  using ClusterBase = OwnedClusterVector<NodeID, NodeID>;

public:
  LPClusteringImpl(const NodeID max_n, const CoarseningContext &c_ctx)
      : ClusterWeightBase(max_n),
        ClusterBase(max_n),
        _c_ctx(c_ctx) {
    allocate(max_n, max_n);
    set_max_degree(c_ctx.lp.large_degree_threshold);
    set_max_num_neighbors(c_ctx.lp.max_num_neighbors);
  }

  void set_max_cluster_weight(const NodeWeight max_cluster_weight) final {
    _max_cluster_weight = max_cluster_weight;
  }

  const AtomicClusterArray &compute_clustering(const Graph &graph) final {
    initialize(&graph, graph.n());

    for (int iteration = 0; iteration < _c_ctx.lp.num_iterations; ++iteration) {
      SCOPED_TIMER("Iteration", std::to_string(iteration));
      if (perform_iteration() == 0) {
        break;
      }
    }

    cluster_isolated_nodes();
    cluster_two_hop_nodes();

    return clusters();
  }

private:
  void cluster_two_hop_nodes() {
    SCOPED_TIMER("Handle two-hop nodes");

    if (!should_handle_two_hop_nodes()) {
      return;
    }

    switch (_c_ctx.lp.two_hop_strategy) {
    case TwoHopStrategy::MATCH:
      match_two_hop_nodes();
      break;

    case TwoHopStrategy::MATCH_THREADWISE:
      match_two_hop_nodes_threadwise();
      break;

    case TwoHopStrategy::CLUSTER:
      cluster_two_hop_nodes();
      break;

    case TwoHopStrategy::CLUSTER_THREADWISE:
      cluster_two_hop_nodes_threadwise();
      break;

    case TwoHopStrategy::LEGACY:
      handle_two_hop_clustering_legacy();
      break;

    case TwoHopStrategy::DISABLE:
      break;
    }
  }

  void cluster_isolated_nodes() {
    SCOPED_TIMER("Handle isolated nodes");

    switch (_c_ctx.lp.isolated_nodes_strategy) {
    case IsolatedNodesClusteringStrategy::MATCH:
      match_isolated_nodes();
      break;

    case IsolatedNodesClusteringStrategy::CLUSTER:
      cluster_isolated_nodes();
      break;

    case IsolatedNodesClusteringStrategy::MATCH_DURING_TWO_HOP:
      if (should_handle_two_hop_nodes()) {
        match_isolated_nodes();
      }
      break;

    case IsolatedNodesClusteringStrategy::CLUSTER_DURING_TWO_HOP:
      if (should_handle_two_hop_nodes()) {
        cluster_isolated_nodes();
      }
      break;

    case IsolatedNodesClusteringStrategy::KEEP:
      break;
    }
  }

  [[nodiscard]] bool should_handle_two_hop_nodes() const {
    return (1.0 - 1.0 * _current_num_clusters / _graph->n()) <= _c_ctx.lp.two_hop_threshold;
  }

  // @todo: old implementation that should no longer be used
  void handle_two_hop_clustering_legacy() {
    // Reset _favored_clusters entries for nodes that are not considered for
    // 2-hop clustering, i.e., nodes that are already clustered with at least one other node or
    // nodes that have more weight than max_weight/2.
    // Set _favored_clusters to dummy entry _graph->n() for isolated nodes
    tbb::parallel_for<NodeID>(0, _graph->n(), [&](const NodeID u) {
      if (u != cluster(u)) {
        _favored_clusters[u] = u;
      } else {
        const auto initial_weight = initial_cluster_weight(u);
        const auto current_weight = cluster_weight(u);
        const auto max_weight = max_cluster_weight(u);
        if (current_weight != initial_weight || current_weight > max_weight / 2) {
          _favored_clusters[u] = u;
        }
      }
    });

    tbb::parallel_for<NodeID>(0, _graph->n(), [&](const NodeID u) {
      // Abort once we have merged enough clusters to achieve the configured minimum shrink factor
      if (should_stop()) {
        return;
      }

      // Skip nodes that should not be considered during 2-hop clustering
      const NodeID favored_leader = _favored_clusters[u];
      if (favored_leader == u) {
        return;
      }

      do {
        // If this works, we set ourself as clustering partners for nodes that have the same favored
        // cluster we have
        NodeID expected_value = favored_leader;
        if (_favored_clusters[favored_leader].compare_exchange_strong(expected_value, u)) {
          break;
        }

        // If this did not work, there is another node that has the same favored cluster
        // Try to join the cluster of that node
        const NodeID partner = expected_value;
        if (_favored_clusters[favored_leader].compare_exchange_strong(
                expected_value, favored_leader
            )) {
          if (move_cluster_weight(u, partner, cluster_weight(u), max_cluster_weight(partner))) {
            move_node(u, partner);
            --_current_num_clusters;
          }

          break;
        }
      } while (true);
    });
  }

public:
  [[nodiscard]] NodeID initial_cluster(const NodeID u) {
    return u;
  }

  [[nodiscard]] NodeWeight initial_cluster_weight(const NodeID cluster) {
    return _graph->node_weight(cluster);
  }

  [[nodiscard]] NodeWeight max_cluster_weight(const NodeID /* cluster */) {
    return _max_cluster_weight;
  }

  [[nodiscard]] bool accept_cluster(const Base::ClusterSelectionState &state) {
    return (state.current_gain > state.best_gain ||
            (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
           (state.current_cluster_weight + state.u_weight <=
                max_cluster_weight(state.current_cluster) ||
            state.current_cluster == state.initial_cluster);
  }

  using Base::_current_num_clusters;
  using Base::_graph;

  const CoarseningContext &_c_ctx;
  NodeWeight _max_cluster_weight = kInvalidBlockWeight;
};

//
// Exposed wrapper
//

LPClustering::LPClustering(const NodeID max_n, const CoarseningContext &c_ctx)
    : _core{std::make_unique<LPClusteringImpl>(max_n, c_ctx)} {}

// we must declare the destructor explicitly here, otherwise, it is implicitly
// generated before LabelPropagationClusterCore is complete
LPClustering::~LPClustering() = default;

void LPClustering::set_max_cluster_weight(const NodeWeight max_cluster_weight) {
  _core->set_max_cluster_weight(max_cluster_weight);
}

void LPClustering::set_desired_cluster_count(const NodeID count) {
  _core->set_desired_num_clusters(count);
}

const Clusterer::AtomicClusterArray &LPClustering::compute_clustering(const Graph &graph) {
  return _core->compute_clustering(graph);
}
} // namespace kaminpar::shm
