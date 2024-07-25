/******************************************************************************
 * Label propagation for graph coarsening / clustering.
 *
 * @file:   legacy_lp_clusterer.cc
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#include "kaminpar-shm/coarsening/clustering/legacy_lp_clusterer.h"

#include <memory>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/legacy_label_propagation.h"

#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
//
// Actual implementation -- not exposed in header
//

struct LegacyLPClusteringConfig : public LegacyLabelPropagationConfig {
  using ClusterID = NodeID;
  using ClusterWeight = BlockWeight;
  static constexpr bool kTrackClusterCount = true;
  static constexpr bool kUseTwoHopClustering = true;
};

class LegacyLPClusteringImpl final
    : public ChunkRandomdLegacyLabelPropagation<LegacyLPClusteringImpl, LegacyLPClusteringConfig>,
      public LegacyOwnedRelaxedClusterWeightVector<NodeID, NodeWeight>,
      public LegacyNonatomicClusterVectorRef<NodeID, NodeID> {
  SET_DEBUG(false);

  using Base = ChunkRandomdLegacyLabelPropagation<LegacyLPClusteringImpl, LegacyLPClusteringConfig>;
  using ClusterWeightBase = LegacyOwnedRelaxedClusterWeightVector<NodeID, NodeWeight>;
  using ClusterBase = LegacyNonatomicClusterVectorRef<NodeID, NodeID>;

public:
  LegacyLPClusteringImpl(const CoarseningContext &c_ctx) : _lp_ctx(c_ctx.clustering.lp) {
    set_max_degree(_lp_ctx.large_degree_threshold);
    set_max_num_neighbors(_lp_ctx.max_num_neighbors);
  }

  void set_max_cluster_weight(const NodeWeight max_cluster_weight) {
    _max_cluster_weight = max_cluster_weight;
  }

  void compute_clustering(StaticArray<NodeID> &clustering, const CSRGraph &graph, bool) {
    KASSERT(clustering.size() >= graph.n(), "preallocated clustering array is too small");

    Base::allocate(graph.n(), graph.n());
    LegacyOwnedRelaxedClusterWeightVector::allocate_cluster_weights(graph.n());
    LegacyNonatomicClusterVectorRef::init_clusters_ref(clustering);
    Base::initialize(&graph, graph.n());

    for (int iteration = 0; iteration < _lp_ctx.num_iterations; ++iteration) {
      SCOPED_TIMER("Iteration", std::to_string(iteration));
      if (perform_iteration() == 0) {
        break;
      }
    }

    cluster_isolated_nodes();
    cluster_two_hop_nodes();
  }

private:
  void cluster_two_hop_nodes() {
    SCOPED_TIMER("Handle two-hop nodes");

    if (!should_handle_two_hop_nodes()) {
      return;
    }

    switch (_lp_ctx.two_hop_strategy) {
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

    switch (_lp_ctx.isolated_nodes_strategy) {
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
    return (1.0 - 1.0 * _current_num_clusters / _graph->n()) <= _lp_ctx.two_hop_threshold;
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
        if (__atomic_compare_exchange_n(
                &_favored_clusters[favored_leader],
                &expected_value,
                u,
                false,
                __ATOMIC_SEQ_CST,
                __ATOMIC_SEQ_CST
            )) {
          break;
        }

        // If this did not work, there is another node that has the same favored cluster
        // Try to join the cluster of that node
        const NodeID partner = expected_value;
        if (__atomic_compare_exchange_n(
                &_favored_clusters[favored_leader],
                &expected_value,
                favored_leader,
                false,
                __ATOMIC_SEQ_CST,
                __ATOMIC_SEQ_CST
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

  const LabelPropagationCoarseningContext &_lp_ctx;
  NodeWeight _max_cluster_weight = kInvalidBlockWeight;
};

//
// Exposed wrapper
//

LegacyLPClustering::LegacyLPClustering(const CoarseningContext &c_ctx)
    : _core(std::make_unique<LegacyLPClusteringImpl>(c_ctx)) {}

// we must declare the destructor explicitly here, otherwise, it is implicitly
// generated before LegacyLabelPropagationClusterCore is complete
LegacyLPClustering::~LegacyLPClustering() = default;

void LegacyLPClustering::set_max_cluster_weight(const NodeWeight max_cluster_weight) {
  _core->set_max_cluster_weight(max_cluster_weight);
}

void LegacyLPClustering::set_desired_cluster_count(const NodeID count) {
  _core->set_desired_num_clusters(count);
}

void LegacyLPClustering::compute_clustering(
    StaticArray<NodeID> &clustering, const Graph &graph, bool
) {
  _core->compute_clustering(clustering, graph.csr_graph(), false);
}
} // namespace kaminpar::shm
