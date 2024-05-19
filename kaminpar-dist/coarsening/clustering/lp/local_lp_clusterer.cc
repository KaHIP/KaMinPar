/*******************************************************************************
 * Label propagation clustering that only clusters node within a PE (i.e.,
 * ignores ghost nodes).
 *
 * @file:   local_lp_clusterer.cc
 * @author: Daniel Seemaier
 * @date:   30.09.2021
 ******************************************************************************/
#include "kaminpar-dist/coarsening/clustering/lp/local_lp_clusterer.h"

#include "kaminpar-dist/distributed_label_propagation.h"

namespace kaminpar::dist {
struct LocalLPClusteringConfig : public LabelPropagationConfig {
  using ClusterID = NodeID;
  using ClusterWeight = NodeWeight;

  static constexpr bool kTrackClusterCount = false;
  static constexpr bool kUseTwoHopClustering = true;
};

class LocalLPClusteringImpl final
    : public ChunkRandomdLabelPropagation<LocalLPClusteringImpl, LocalLPClusteringConfig>,
      public NonatomicClusterVectorRef<NodeID, NodeID>,
      public OwnedRelaxedClusterWeightVector<NodeID, NodeWeight> {
  SET_DEBUG(false);

  using Base = ChunkRandomdLabelPropagation<LocalLPClusteringImpl, LocalLPClusteringConfig>;
  using ClusterBase = NonatomicClusterVectorRef<NodeID, NodeID>;
  using ClusterWeightBase = OwnedRelaxedClusterWeightVector<NodeID, NodeWeight>;

public:
  LocalLPClusteringImpl(const NodeID max_n, const CoarseningContext &c_ctx)
      : _ignore_ghost_nodes(c_ctx.local_lp.ignore_ghost_nodes),
        _keep_ghost_clusters(c_ctx.local_lp.keep_ghost_clusters) {
    allocate_cluster_weights(max_n);
    allocate(max_n, max_n);
    set_max_num_iterations(c_ctx.local_lp.num_iterations);
    set_max_degree(c_ctx.local_lp.active_high_degree_threshold);
    set_max_num_neighbors(c_ctx.local_lp.max_num_neighbors);
  }

  void initialize(const DistributedGraph &graph) {
    Base::initialize(&graph, graph.n());
  }

  void set_max_cluster_weight(const GlobalNodeWeight max_cluster_weight) {
    _max_cluster_weight = max_cluster_weight;
  }

  void compute_clustering(StaticArray<NodeID> &clustering, const DistributedGraph &graph) {
    init_clusters_ref(clustering);
    initialize(graph);

    // initialize ghost nodes
    if (!_ignore_ghost_nodes) {
      init_ghost_nodes();
    }

    DBG << "Computing clustering on graph with " << graph.global_n()
        << " nodes (local: " << graph.n() << ", ghost: " << graph.ghost_n()
        << "), max cluster weight " << _max_cluster_weight << ", and at most "
        << _max_num_iterations << " iterations";

    std::size_t iteration;
    for (iteration = 0; iteration < _max_num_iterations; ++iteration) {
      if (perform_iteration() == 0) {
        break;
      }
    }
    DBG << "Converged / stopped after " << iteration << " iterations";

    // dissolve all clusters owned by ghost nodes
    if (!_ignore_ghost_nodes) {
      if (_keep_ghost_clusters) {
        for (NodeID u : _graph->nodes()) {
          const ClusterID u_cluster = cluster(u);
          if (_graph->is_ghost_node(u_cluster)) {
            // abuse cluster(u_cluster) to remap the whole cluster
            if (_graph->is_ghost_node(cluster(u_cluster))) {
              move_node(u_cluster, u);
            }

            move_node(u, cluster(u_cluster));
          }
        }
      } else {
        graph.pfor_nodes([&](const NodeID u) {
          if (_graph->is_ghost_node(cluster(u))) {
            move_node(u, u);
          }
        });
      }
    }
  }

  void set_max_num_iterations(const std::size_t max_num_iterations) {
    _max_num_iterations =
        (max_num_iterations == 0) ? std::numeric_limits<std::size_t>::max() : max_num_iterations;
  }

  void init_ghost_nodes() {
    tbb::parallel_for(_graph->n(), _graph->total_n(), [&](const NodeID ghost) {
      const ClusterID cluster = initial_cluster(ghost);
      init_cluster(ghost, cluster);
      init_cluster_weight(ghost, initial_cluster_weight(cluster));
    });
  }

  //
  // Called from base class
  //

  [[nodiscard]] NodeID initial_cluster(const NodeID u) {
    return u;
  }

  [[nodiscard]] NodeWeight initial_cluster_weight(const NodeID cluster) {
    return _graph->node_weight(cluster);
  }

  [[nodiscard]] NodeWeight max_cluster_weight(const NodeID) {
    return _max_cluster_weight;
  }

  [[nodiscard]] bool accept_cluster(const Base::ClusterSelectionState &state) {
    return (state.current_gain > state.best_gain ||
            (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
           (state.current_cluster_weight + state.u_weight <=
                max_cluster_weight(state.current_cluster) ||
            state.current_cluster == state.initial_cluster);
  }

  [[nodiscard]] bool accept_neighbor(const NodeID u, const NodeID v) {
    return (_partition == nullptr || _partition[u] == _partition[v]) &&
           (_ignore_ghost_nodes == false || _graph->is_owned_node(v));
  }

  [[nodiscard]] bool activate_neighbor(const NodeID u) {
    return _graph->is_owned_node(u);
  }

  using Base::_graph;
  NodeWeight _max_cluster_weight = std::numeric_limits<NodeWeight>::max();
  std::size_t _max_num_iterations;
  bool _ignore_ghost_nodes;
  bool _keep_ghost_clusters;

  const BlockID *_partition = nullptr;
};

//
// Interface
//

LocalLPClusterer::LocalLPClusterer(const Context &ctx)
    : _impl(std::make_unique<LocalLPClusteringImpl>(
          ctx.coarsening.local_lp.ignore_ghost_nodes ? ctx.partition.graph->n
                                                     : ctx.partition.graph->total_n,
          ctx.coarsening
      )) {}

LocalLPClusterer::~LocalLPClusterer() = default;

void LocalLPClusterer::set_communities(const StaticArray<BlockID> &communities) {
  _impl->_partition = communities.data();
}

void LocalLPClusterer::clear_communities() {
  _impl->_partition = nullptr;
}

void LocalLPClusterer::set_max_cluster_weight(GlobalNodeWeight weight) {
  _impl->set_max_cluster_weight(weight);
}

void LocalLPClusterer::cluster(
    StaticArray<GlobalNodeID> &global_clustering, const DistributedGraph &p_graph
) {
  StaticArray<NodeID> local_clustering(
      p_graph.n(), reinterpret_cast<NodeID *>(global_clustering.data())
  );
  return _impl->compute_clustering(local_clustering, p_graph);
}
} // namespace kaminpar::dist
