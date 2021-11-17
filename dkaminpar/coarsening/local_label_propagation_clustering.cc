/*******************************************************************************
 * @file:   distributed_local_label_propagation_clustering.cc
 *
 * @author: Daniel Seemaier
 * @date:   30.09.21
 * @brief:
 ******************************************************************************/
#include "dkaminpar/coarsening/local_label_propagation_clustering.h"

#include "kaminpar/label_propagation.h"

namespace dkaminpar {
struct DistributedLocalLabelPropagationClusteringConfig : public shm::LabelPropagationConfig {
  using Graph = DistributedGraph;
  using ClusterID = NodeID;
  using ClusterWeight = NodeWeight;
  static constexpr bool kTrackClusterCount = false;
  static constexpr bool kUseTwoHopClustering = true;
};

class DistributedLocalLabelPropagationClusteringImpl final
    : public shm::ChunkRandomizedLabelPropagation<DistributedLocalLabelPropagationClusteringImpl,
                                                  DistributedLocalLabelPropagationClusteringConfig>,
      public shm::OwnedClusterVector<NodeID, NodeID>,
      public shm::OwnedRelaxedClusterWeightVector<NodeID, NodeWeight> {
  SET_DEBUG(true);

  using Base = shm::ChunkRandomizedLabelPropagation<DistributedLocalLabelPropagationClusteringImpl,
                                                    DistributedLocalLabelPropagationClusteringConfig>;
  using ClusterBase = shm::OwnedClusterVector<NodeID, NodeID>;
  using ClusterWeightBase = shm::OwnedRelaxedClusterWeightVector<NodeID, NodeWeight>;

  static constexpr auto kInfiniteIterations = std::numeric_limits<std::size_t>::max();

public:
  DistributedLocalLabelPropagationClusteringImpl(const NodeID max_n, const CoarseningContext &c_ctx)
      : ClusterBase{max_n}, ClusterWeightBase{max_n} {
    allocate(max_n);
    set_max_num_iterations(c_ctx.local_lp.num_iterations);
    set_max_degree(c_ctx.local_lp.large_degree_threshold);
    set_max_num_neighbors(c_ctx.local_lp.max_num_neighbors);
  }

  const auto &compute_clustering(const DistributedGraph &graph, const NodeWeight max_cluster_weight,
                                 const std::size_t max_iterations = kInfiniteIterations) {
    initialize(&graph, graph.n());
    _max_cluster_weight = max_cluster_weight;

    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
      if (perform_iteration() == 0) {
        break;
      }
    }

    return clusters();
  }

  void set_max_num_iterations(const std::size_t max_num_iterations) {
    _max_num_iterations = (max_num_iterations == 0) ? std::numeric_limits<std::size_t>::max() : max_num_iterations;
  }

  //
  // Called from base class
  //

  [[nodiscard]] NodeID initial_cluster(const NodeID u) { return u; }

  [[nodiscard]] NodeWeight initial_cluster_weight(const NodeID cluster) { return _graph->node_weight(cluster); }

  [[nodiscard]] NodeWeight max_cluster_weight(const NodeID) { return _max_cluster_weight; }

  [[nodiscard]] bool accept_cluster(const Base::ClusterSelectionState &state) {
    return (state.current_gain > state.best_gain ||
            (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
           (state.current_cluster_weight + state.u_weight < max_cluster_weight(state.current_cluster) ||
            state.current_cluster == state.initial_cluster);
  }

  [[nodiscard]] bool accept_neighbor(const NodeID u) { return _graph->is_owned_node(u); }

  [[nodiscard]] bool activate_neighbor(const NodeID u) { return _graph->is_owned_node(u); }

  using Base::_graph;
  NodeWeight _max_cluster_weight;
  std::size_t _max_num_iterations;
};

//
// Interface
//

DistributedLocalLabelPropagationClustering::DistributedLocalLabelPropagationClustering(const NodeID max_n,
                                                                                       const CoarseningContext &c_ctx)
    : _impl{std::make_unique<DistributedLocalLabelPropagationClusteringImpl>(max_n, c_ctx)} {}

DistributedLocalLabelPropagationClustering::~DistributedLocalLabelPropagationClustering() = default;

const DistributedLocalLabelPropagationClustering::AtomicClusterArray &
DistributedLocalLabelPropagationClustering::compute_clustering(const DistributedGraph &graph,
                                                               NodeWeight max_cluster_weight) {
  return _impl->compute_clustering(graph, max_cluster_weight);
}
} // namespace dkaminpar