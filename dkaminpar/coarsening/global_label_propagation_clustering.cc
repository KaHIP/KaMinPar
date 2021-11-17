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
#include "kaminpar/label_propagation.h"

#include <unordered_map>

namespace dkaminpar {
namespace {
/*!
 * Large rating map based on a \c unordered_map. We need this since cluster IDs are global node IDs.
 */
struct UnorderedRatingMap {
  EdgeWeight &operator[](const GlobalNodeID key) { return map[key]; }
  [[nodiscard]] auto &entries() { return map; }
  void clear() { map.clear(); }
  std::size_t capacity() const { return std::numeric_limits<std::size_t>::max(); }
  void resize(const std::size_t /* capacity */) {}
  std::unordered_map<GlobalNodeID, EdgeWeight> map{};
};

struct DistributedGlobalLabelPropagationClusteringConfig : public shm::LabelPropagationConfig {
  using Graph = DistributedGraph;
  using RatingMap = ::kaminpar::RatingMap<EdgeWeight, UnorderedRatingMap>;
  using ClusterID = GlobalNodeID;
  using ClusterWeight = GlobalNodeWeight;
  static constexpr bool kTrackClusterCount = false;
  static constexpr bool kUseTwoHopClustering = false;
};
} // namespace

class DistributedGlobalLabelPropagationClusteringImpl final
    : public shm::InOrderLabelPropagation<DistributedGlobalLabelPropagationClusteringImpl,
                                          DistributedGlobalLabelPropagationClusteringConfig>,
      public shm::OwnedClusterVector<NodeID, GlobalNodeID> {
  SET_DEBUG(false);

  using Base = shm::InOrderLabelPropagation<DistributedGlobalLabelPropagationClusteringImpl,
                                            DistributedGlobalLabelPropagationClusteringConfig>;
  using ClusterBase = shm::OwnedClusterVector<NodeID, GlobalNodeID>;

public:
  DistributedGlobalLabelPropagationClusteringImpl(const Context &ctx)
      : ClusterBase{ctx.partition.total_n()}, _c_ctx{ctx.coarsening},
        _changed_label(ctx.partition.local_n()), _cluster_weights{ctx.partition.local_n()} {
    set_max_num_iterations(_c_ctx.global_lp.num_iterations);
    set_max_degree(_c_ctx.global_lp.large_degree_threshold);
    set_max_num_neighbors(_c_ctx.global_lp.max_num_neighbors);
  }

  const auto &compute_clustering(const DistributedGraph &graph, const NodeWeight max_cluster_weight) {
    TIMED_SCOPE("Allocation") { allocate(graph); };
    initialize(&graph, graph.total_n());
    initialize_ghost_node_clusters();
    _max_cluster_weight = max_cluster_weight;

    for (std::size_t iteration = 0; iteration < _max_num_iterations; ++iteration) {
      GlobalNodeID global_num_moved_nodes = 0;
      for (std::size_t chunk = 0; chunk < _c_ctx.global_lp.num_chunks; ++chunk) {
        const auto [from, to] = math::compute_local_range<NodeID>(_graph->n(), _c_ctx.global_lp.num_chunks, chunk);
        global_num_moved_nodes += process_chunk(from, to);
      }
      if (global_num_moved_nodes == 0) {
        break;
      }
    }

    return clusters();
  }

  void set_max_num_iterations(const std::size_t max_num_iterations) {
    _max_num_iterations = max_num_iterations == 0 ? std::numeric_limits<std::size_t>::max() : max_num_iterations;
  }

  //--------------------------------------------------------------------------------
  //
  // Called from base class
  //
  // VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
  void reset_node_state(const NodeID u) {
    Base::reset_node_state(u);
    _changed_label[u] = 0;
  }

  /*
   * Cluster weights
   *
   * Note: offset cluster IDs by 1 since growt cannot use 0 as key
   */

  void init_cluster_weight(const GlobalNodeID local_cluster, const NodeWeight weight) {
    ASSERT(local_cluster < _graph->total_n());
    const auto cluster = _graph->local_to_global_node(static_cast<NodeID>(local_cluster));

    auto &handle = _cluster_weights_handles_ets.local();
    [[maybe_unused]] const auto [it, success] = handle.insert(cluster + 1, weight);
    ASSERT(success);
  }

  NodeWeight cluster_weight(const GlobalNodeID cluster) {
    auto &handle = _cluster_weights_handles_ets.local();
    auto it = handle.find(cluster + 1);
    ASSERT(it != handle.end()) << "Uninitialized cluster: " << cluster + 1;

    return (*it).second;
  }

  bool move_cluster_weight(const GlobalNodeID old_cluster, const GlobalNodeID new_cluster, const NodeWeight delta,
                           const NodeWeight max_weight) {
    if (cluster_weight(new_cluster) + delta <= max_weight) {
      auto &handle = _cluster_weights_handles_ets.local();
      [[maybe_unused]] const auto [old_it, old_found] = handle.update(
          old_cluster + 1, [](auto &lhs, const auto rhs) { return lhs -= rhs; }, delta);
      ASSERT(old_it != handle.end() && old_found) << "Uninitialized cluster: " << old_cluster + 1;

      [[maybe_unused]] const auto [new_it, new_found] = handle.update(
          new_cluster + 1, [](auto &lhs, const auto rhs) { return lhs += rhs; }, delta);
      ASSERT(new_it != handle.end() && new_found) << "Uninitialized cluster: " << new_cluster + 1;

      return true;
    }
    return false;
  }

  void change_cluster_weight(const GlobalNodeID cluster, const NodeWeight delta,
                             [[maybe_unused]] const bool must_exist) {
    auto &handle = _cluster_weights_handles_ets.local();
    [[maybe_unused]] const auto [it, not_found] = handle.insert_or_update(
        cluster + 1, delta, [](auto &lhs, const auto rhs) { return lhs += rhs; }, delta);
    ASSERT(it != handle.end() && (!must_exist || !not_found)) << "Could not update cluster: " << cluster;
  }

  [[nodiscard]] NodeWeight initial_cluster_weight(const GlobalNodeID u) {
    ASSERT(u < _graph->total_n());
    return _graph->node_weight(static_cast<NodeID>(u));
  }

  [[nodiscard]] NodeWeight max_cluster_weight(const GlobalNodeID /* cluster */) { return _max_cluster_weight; }

  [[nodiscard]] NodeWeight initial_cluster_weight(const NodeID u) { return _graph->node_weight(u); }

  /*
   * Clusters
   */

  void move_node(const NodeID node, const GlobalNodeID cluster) {
    OwnedClusterVector::move_node(node, cluster);
    _changed_label[node] = 1;
  }

  [[nodiscard]] GlobalNodeID initial_cluster(const NodeID u) { return _graph->local_to_global_node(u); }

  /*
   * Moving nodes
   */

  [[nodiscard]] bool accept_cluster(const Base::ClusterSelectionState &state) {
    return (state.current_gain > state.best_gain ||
            (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
           (state.current_cluster_weight + state.u_weight < max_cluster_weight(state.current_cluster) ||
            state.current_cluster == state.initial_cluster);
  }

  [[nodiscard]] inline bool activate_neighbor(const NodeID u) { return _graph->is_owned_node(u); }
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  //
  // Called from base class
  //
  //--------------------------------------------------------------------------------

private:
  void allocate(const DistributedGraph &graph) {
    ensure_cluster_size(graph.total_n());

    const NodeID allocated_num_active_nodes = _changed_label.size();

    if (allocated_num_active_nodes < graph.n()) {
      _changed_label.resize(graph.n());
    }

    Base::allocate(graph.total_n(), graph.n());
  }

  void initialize_ghost_node_clusters() {
    tbb::parallel_for(_graph->n(), _graph->total_n(), [&](const NodeID local_u) {
      const GlobalNodeID label = _graph->local_to_global_node(local_u);
      init_cluster(local_u, label);
    });
  }

  GlobalNodeID process_chunk(const NodeID from, const NodeID to) {
    DBG << "process_chunk(" << from << ".." << to << ")";

    const NodeID local_num_moved_nodes = perform_iteration(from, to);
    const GlobalNodeID global_num_moved_nodes = mpi::allreduce(local_num_moved_nodes, MPI_SUM, _graph->communicator());

    if (global_num_moved_nodes > 0) {
      synchronize_ghost_node_clusters(from, to);
    }

    return global_num_moved_nodes;
  }

  void synchronize_ghost_node_clusters(const NodeID from, const NodeID to) {
    struct ChangedLabelMessage {
      NodeID local_node;
      GlobalNodeID new_label;
    };

    mpi::graph::sparse_alltoall_interface_to_pe<ChangedLabelMessage, scalable_vector>(
        *_graph, from, to, [&](const NodeID u) { return _changed_label[u]; },
        [&](const NodeID u) -> ChangedLabelMessage {
          return {u, cluster(u)};
        },
        [&](const auto &buffer, const PEID pe) {
          tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
            const auto [local_node_on_pe, new_label] = buffer[i];

            const GlobalNodeID global_node = _graph->offset_n(pe) + local_node_on_pe;
            ASSERT(!_graph->is_owned_global_node(global_node));

            const NodeID local_node = _graph->global_to_local_node(global_node);
            const NodeWeight local_node_weight = _graph->node_weight(local_node);

            change_cluster_weight(cluster(local_node), -local_node_weight, true);
            OwnedClusterVector::move_node(local_node, new_label);
            change_cluster_weight(cluster(local_node), local_node_weight, false);
          });
        });
  }

  using Base::_graph;
  const CoarseningContext &_c_ctx;
  NodeWeight _max_cluster_weight{std::numeric_limits<NodeWeight>::max()};
  std::size_t _max_num_iterations{std::numeric_limits<std::size_t>::max()};

  //! \code{_changed_label[u] = 1} iff. node \c u changed its label in the current round
  scalable_vector<uint8_t> _changed_label;

  using ClusterWeightsMap = typename growt::GlobalNodeIDMap<GlobalNodeWeight>;
  ClusterWeightsMap _cluster_weights{0};
  tbb::enumerable_thread_specific<typename ClusterWeightsMap::handle_type> _cluster_weights_handles_ets{
      [&] { return ClusterWeightsMap::handle_type{_cluster_weights}; }};
};

//
// Exposed wrapper
//

DistributedGlobalLabelPropagationClustering::DistributedGlobalLabelPropagationClustering(const Context &ctx)
    : _impl{std::make_unique<DistributedGlobalLabelPropagationClusteringImpl>(ctx)} {}

DistributedGlobalLabelPropagationClustering::~DistributedGlobalLabelPropagationClustering() = default;

const DistributedGlobalLabelPropagationClustering::AtomicClusterArray &
DistributedGlobalLabelPropagationClustering::compute_clustering(const DistributedGraph &graph,
                                                                const NodeWeight max_cluster_weight) {
  return _impl->compute_clustering(graph, max_cluster_weight);
}
} // namespace dkaminpar