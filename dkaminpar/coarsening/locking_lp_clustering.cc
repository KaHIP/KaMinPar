/*******************************************************************************
 * @file:   locking_lp_clustering.cc
 *
 * @author: Daniel Seemaier
 * @date:   01.10.21
 * @brief:
 ******************************************************************************/
#include "dkaminpar/coarsening/locking_lp_clustering.h"

#include "dkaminpar/growt.h"
#include "dkaminpar/mpi_graph_utils.h"
#include "dkaminpar/utility/distributed_math.h"
#include "kaminpar/algorithm/parallel_label_propagation.h"

namespace dkaminpar {
namespace {
struct LockingLpClusteringConfig : shm::LabelPropagationConfig {
  using Graph = DistributedGraph;
  using ClusterID = NodeID;
  using ClusterWeight = NodeWeight;
};

template<typename ClusterID, typename ClusterWeight>
class OwnedRelaxedClusterWeightMap {
  using hasher_type = utils_tm::hash_tm::murmur2_hash;
  using allocator_type = growt::AlignedAllocator<>;
  using table_type = typename growt::table_config<ClusterID, ClusterWeight, hasher_type, allocator_type, hmod::growable,
                                                  hmod::deletion>::table_type;

public:
  explicit OwnedRelaxedClusterWeightMap(const ClusterID max_num_clusters) : _cluster_weights(max_num_clusters) {}

  void init_cluster_weight(const ClusterID cluster, const ClusterWeight weight) {
    _cluster_weights_handles_ets.local().insert(cluster, weight);
  }

  ClusterWeight cluster_weight(const ClusterID cluster) {
    auto &handle = _cluster_weights_handles_ets.local();
    auto it = handle.find(cluster);
    ASSERT(it != handle.end());
    return (*it).second;
  }

  bool move_cluster_weight(const ClusterID old_cluster, const ClusterID new_cluster, const ClusterWeight delta,
                           const ClusterWeight max_weight) {
    if (cluster_weight(old_cluster) + delta <= max_weight) {
      auto &handle = _cluster_weights_handles_ets.local();
      // clang-format off
      handle.update(old_cluster, [delta](auto &lhs, const auto rhs) { return lhs -= rhs; }, delta);
      handle.update(new_cluster, [delta](auto &lhs, const auto rhs) { return lhs += rhs; }, delta);
      // clang-format on
      return true;
    }
    return false;
  }

private:
  table_type _cluster_weights;
  tbb::enumerable_thread_specific<typename table_type::handle_type> _cluster_weights_handles_ets{
      [&] { return _cluster_weights.get_handle(); }};
};
} // namespace

class LockingLpClusteringImpl : public shm::InOrderLabelPropagation<LockingLpClusteringImpl, LockingLpClusteringConfig>,
                                public OwnedRelaxedClusterWeightMap<GlobalNodeID, NodeWeight> {
  using Base = shm::InOrderLabelPropagation<LockingLpClusteringImpl, LockingLpClusteringConfig>;
  using ClusterWeightBase = OwnedRelaxedClusterWeightMap<GlobalNodeID, NodeWeight>;
  using AtomicClusterArray = scalable_vector<shm::parallel::IntegralAtomicWrapper<GlobalNodeID>>;

  friend Base;
  friend Base::Base;

public:
  LockingLpClusteringImpl(const NodeID max_n, const CoarseningContext &c_ctx)
      : Base{max_n},
        ClusterWeightBase{max_n},
        _c_ctx{c_ctx},
        _current_clustering(max_n),
        _next_clustering(max_n) {
    set_max_degree(c_ctx.lp.large_degree_threshold);
    set_max_num_neighbors(c_ctx.lp.max_num_neighbors);
  }

  const auto &compute_clustering(const DistributedGraph &graph, const NodeWeight max_cluster_weight) {
    initialize(&graph, graph.total_n());
    _max_cluster_weight = max_cluster_weight;

    const auto num_iterations = _c_ctx.lp.num_iterations == 0 ? std::numeric_limits<std::size_t>::max()
                                                              : _c_ctx.lp.num_iterations;

    for (std::size_t iteration = 0; iteration < num_iterations; ++iteration) {
      NodeID num_moved_nodes = 0;
      for (std::size_t chunk = 0; chunk < _c_ctx.lp.num_chunks; ++chunk) {
        const auto [from, to] = math::compute_local_range<NodeID>(_graph->n(), _c_ctx.lp.num_chunks, chunk);
        num_moved_nodes += process_chunk(from, to);
      }
      if (num_moved_nodes == 0) { break; }
    }

    return _current_clustering;
  }

protected:
  //--------------------------------------------------------------------------------
  // Called from base class
  //VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
  void init_cluster(const NodeID node, const NodeID cluster) {
    _current_clustering[node] = cluster;
    _next_clustering[node] = cluster;
  }

  [[nodiscard]] NodeID cluster(const NodeID u) const { return _next_clustering[u]; }

  void move_node(const NodeID node, const GlobalNodeID cluster) { _next_clustering[node] = cluster; }

  [[nodiscard]] NodeID initial_cluster(const NodeID u) const { return u; }

  [[nodiscard]] NodeWeight initial_cluster_weight(const NodeID u) const { return _graph->node_weight(u); }

  [[nodiscard]] NodeWeight max_cluster_weight(const GlobalNodeID /* cluster */) const { return _max_cluster_weight; }

  [[nodiscard]] bool accept_cluster(const Base::ClusterSelectionState &state) const {
    return (state.current_gain > state.best_gain ||
            (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
           (state.current_cluster_weight + state.u_weight < max_cluster_weight(state.current_cluster) ||
            state.current_cluster == state.initial_cluster);
  }

  [[nodiscard]] inline bool activate_neighbor(const NodeID u) const { return _graph->is_owned_node(u); }
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // Called from base class
  //--------------------------------------------------------------------------------

private:
  /*
   * TODO: deactivate locked nodes
   */
  NodeID process_chunk(const NodeID from, const NodeID to) {
    const NodeID num_moved_nodes = perform_iteration(from, to);
    if (num_moved_nodes == 0) { return 0; } // nothing to do

    perform_distributed_moves(from, to);

    return num_moved_nodes;
  }

  void perform_distributed_moves(const NodeID from, const NodeID to) {
    struct JoinRequest {
      GlobalNodeID global_requester;
      NodeID requester_weight;
      EdgeWeight requester_gain;
      GlobalNodeID global_requested;
    };

    struct JoinResponse {
      GlobalNodeID global_requested;
      NodeWeight new_weight;
      std::uint8_t response;
    };


  }

  //! Synchronize labels of ghost nodes.
  // TODO: have to use global labels
  void synchronize_labels(const NodeID from, const NodeID to) {
    struct LabelMessage {
      GlobalNodeID global_node;
      GlobalNodeID global_new_label;
    };

    mpi::graph::sparse_alltoall_interface_to_pe<LabelMessage>(
        *_graph, from, to, [&](const NodeID u) { return _next_clustering[u] != _current_clustering[u]; },
        [&](const NodeID u) -> LabelMessage {
          return {_graph->local_to_global_node(u), _next_clustering[u]};
        },
        [&](const auto buffer) {
          tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
            const auto [global_node, global_new_label] = buffer[i];
            const auto local_node = _graph->global_to_local_node(global_node);
            move_node(local_node, global_new_label);
          });
        });
  }

  using Base::_graph;

  const CoarseningContext &_c_ctx;

  NodeWeight _max_cluster_weight;
  AtomicClusterArray _current_clustering;
  AtomicClusterArray _next_clustering;
};

LockingLpClustering::LockingLpClustering(const NodeID max_n, const CoarseningContext &c_ctx)
    : _impl{std::make_unique<LockingLpClusteringImpl>(max_n, c_ctx)} {}

LockingLpClustering::~LockingLpClustering() = default;

const LockingLpClustering::AtomicClusterArray &
LockingLpClustering::compute_clustering(const DistributedGraph &graph, const NodeWeight max_cluster_weight) {
  return _impl->compute_clustering(graph, max_cluster_weight);
}
} // namespace dkaminpar