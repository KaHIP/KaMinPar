/*******************************************************************************
 * @file:   global_label_propagation_coarsener.cc
 *
 * @author: Daniel Seemaier
 * @date:   29.09.21
 * @brief:  Label propagation across PEs, synchronizes cluster weights after
 * every weight, otherwise moves nodes without communication causing violations
 * of the balance constraint.
 ******************************************************************************/
#include "dkaminpar/coarsening/global_label_propagation_clustering.h"

#include "dkaminpar/growt.h"
#include "dkaminpar/mpi_graph.h"
#include "dkaminpar/utility/math.h"
#include "kaminpar/algorithm/parallel_label_propagation.h"

namespace dkaminpar {
template <typename ClusterID, typename ClusterWeight> class OwnedRelaxedClusterWeightMap {
  using hasher_type = ::utils_tm::hash_tm::murmur2_hash;
  using allocator_type = ::growt::AlignedAllocator<>;
  using table_type = typename ::growt::table_config<ClusterID, ClusterWeight, hasher_type, allocator_type,
                                                    hmod::growable, hmod::deletion>::table_type;

public:
  explicit OwnedRelaxedClusterWeightMap(const ClusterID max_num_clusters) : _cluster_weights(max_num_clusters) {}

  auto &&take_cluster_weights() { return std::move(_cluster_weights); }

  void init_cluster_weight(const ClusterID cluster, const ClusterWeight weight) {
    _cluster_weights.get_handle().insert(cluster, weight);
  }

  ClusterWeight cluster_weight(const ClusterID cluster) /* const */ {
    auto &handle = _cluster_weights_handles_ets.local();
    auto it = handle.find(cluster);
    return (*it).second;
  }

  bool move_cluster_weight(const ClusterID old_cluster, const ClusterID new_cluster, const ClusterWeight delta,
                           const ClusterWeight max_weight) {
    if (cluster_weight(old_cluster) + delta <= max_weight) {
      auto &handle = _cluster_weights_handles_ets.local();

      const auto [old_it, old_found] = handle.update(
          old_cluster, [delta](auto &lhs, const auto rhs) { return lhs -= rhs; }, delta);
      const auto [new_it, new_found] = handle.update(
          new_cluster, [delta](auto &lhs, const auto rhs) { return lhs += rhs; }, delta);

      ASSERT(old_found);
      UNUSED(old_found);
      ASSERT(new_found);
      UNUSED(new_found);

      return true;
    }
    return false;
  }

private:
  table_type _cluster_weights;
  tbb::enumerable_thread_specific<typename table_type::handle_type> _cluster_weights_handles_ets{
      [&] { return _cluster_weights.get_handle(); }};
};

struct DistributedGlobalLabelPropagationClusteringConfig : public shm::LabelPropagationConfig {
  using Graph = DistributedGraph;
  using ClusterID = GlobalNodeID;
  using ClusterWeight = GlobalNodeWeight;
  static constexpr bool kTrackClusterCount = false;
  static constexpr bool kUseTwoHopClustering = true;
};

class DistributedGlobalLabelPropagationClusteringImpl final
    : public shm::InOrderLabelPropagation<DistributedGlobalLabelPropagationClusteringImpl,
                                          DistributedGlobalLabelPropagationClusteringConfig>,
      public OwnedRelaxedClusterWeightMap<GlobalNodeID, NodeWeight>,
      public shm::OwnedClusterVector<NodeID, GlobalNodeID> {
  SET_DEBUG(true);

  using Base = shm::InOrderLabelPropagation<DistributedGlobalLabelPropagationClusteringImpl,
                                            DistributedGlobalLabelPropagationClusteringConfig>;
  using ClusterWeightBase = OwnedRelaxedClusterWeightMap<GlobalNodeID, NodeWeight>;
  using ClusterBase = shm::OwnedClusterVector<NodeID, GlobalNodeID>;

public:
  DistributedGlobalLabelPropagationClusteringImpl(const NodeID max_n, const CoarseningContext &c_ctx)
      : ClusterWeightBase{max_n}, ClusterBase{max_n}, _c_ctx{c_ctx}, _changed_label(max_n) {
    allocate(max_n);
    set_max_num_iterations(c_ctx.lp.num_iterations);
    set_max_degree(c_ctx.lp.large_degree_threshold);
    set_max_num_neighbors(c_ctx.lp.max_num_neighbors);
  }

  const auto &compute_clustering(const DistributedGraph &graph, const NodeWeight max_cluster_weight) {
    initialize(&graph, graph.total_n());
    _max_cluster_weight = max_cluster_weight;

    for (std::size_t iteration = 0; iteration < _max_num_iterations; ++iteration) {
      NodeID num_moved_nodes = 0;
      for (std::size_t chunk = 0; chunk < _c_ctx.lp.num_chunks; ++chunk) {
        const auto [from, to] = math::compute_local_range<NodeID>(_graph->n(), _c_ctx.lp.num_chunks, chunk);
        num_moved_nodes += process_chunk(from, to);
      }
      if (num_moved_nodes == 0) {
        break;
      }
    }

    return clusters();
  }

  void set_max_num_iterations(const std::size_t max_num_iterations) {
    _max_num_iterations = max_num_iterations == 0 ? std::numeric_limits<std::size_t>::max() : max_num_iterations;
  }

  void init_cluster(const NodeID node, const GlobalNodeID cluster) override {
    OwnedClusterVector::init_cluster(node, cluster);
    _changed_label[node] = 0;
  }
  void move_node(const NodeID node, const GlobalNodeID cluster) override {
    OwnedClusterVector::move_node(node, cluster);
    _changed_label[node] = 1;
  }

  [[nodiscard]] GlobalNodeID initial_cluster(const NodeID u) const { return _graph->local_to_global_node(u); }

  [[nodiscard]] NodeWeight initial_cluster_weight(const GlobalNodeID cluster) const {
    return _graph->node_weight(_graph->global_to_local_node(cluster));
  }

  [[nodiscard]] NodeWeight max_cluster_weight(const GlobalNodeID /* cluster */) const { return _max_cluster_weight; }

  [[nodiscard]] bool accept_cluster(const Base::ClusterSelectionState &state) const {
    return (state.current_gain > state.best_gain ||
            (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
           (state.current_cluster_weight + state.u_weight < max_cluster_weight(state.current_cluster) ||
            state.current_cluster == state.initial_cluster);
  }

  [[nodiscard]] inline bool activate_neighbor(const NodeID u) const { return _graph->is_owned_node(u); }

private:
  NodeID process_chunk(const NodeID from, const NodeID to) {
    const NodeID num_moved_nodes = perform_iteration(from, to);
    if (num_moved_nodes == 0) {
      return 0;
    } // nothing to do

    synchronize_ghost_node_clusters(from, to);
    return num_moved_nodes;
  }

  void synchronize_ghost_node_clusters(const NodeID from, const NodeID to) {
    struct ChangedLabelMessage {
      GlobalNodeID global_node;
      GlobalNodeID new_label;
      NodeWeight new_cluster_weight;
    };

    mpi::graph::sparse_alltoall_interface_to_pe<ChangedLabelMessage, scalable_vector>(
        *_graph, from, to, [&](const NodeID u) { return _changed_label[u]; },
        [&](const NodeID u) -> ChangedLabelMessage {
          const auto leader = cluster(u);
          return {_graph->local_to_global_node(u), leader, cluster_weight(leader)};
        },
        [&](const auto &buffer) {
          tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
            const auto [global_node, new_label, new_cluster_weight] = buffer[i];
            const NodeID local_node = _graph->global_to_local_node(global_node);
            init_cluster_weight(new_label, new_cluster_weight);
            move_node(local_node, new_label);
          });
        });
  }

  using Base::_graph;
  const CoarseningContext &_c_ctx;
  NodeWeight _max_cluster_weight{std::numeric_limits<NodeWeight>::max()};
  std::size_t _max_num_iterations{std::numeric_limits<std::size_t>::max()};

  //! \code{_changed_label[u] = 1} iff. node \c u changed its label in the current round
  scalable_vector<uint8_t> _changed_label;
};

//
// Exposed wrapper
//

DistributedGlobalLabelPropagationClustering::DistributedGlobalLabelPropagationClustering(const NodeID max_n,
                                                                                         const CoarseningContext &c_ctx)
    : _impl{std::make_unique<DistributedGlobalLabelPropagationClusteringImpl>(max_n, c_ctx)} {}

DistributedGlobalLabelPropagationClustering::~DistributedGlobalLabelPropagationClustering() = default;

const DistributedGlobalLabelPropagationClustering::AtomicClusterArray &
DistributedGlobalLabelPropagationClustering::compute_clustering(const DistributedGraph &graph,
                                                                const NodeWeight max_cluster_weight) {
  return _impl->compute_clustering(graph, max_cluster_weight);
}
} // namespace dkaminpar