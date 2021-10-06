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
  using ClusterID = GlobalNodeID;
  using ClusterWeight = NodeWeight;
};
} // namespace

class LockingLpClusteringImpl
    : public shm::InOrderLabelPropagation<LockingLpClusteringImpl, LockingLpClusteringConfig> {
  SET_DEBUG(true);
  SET_STATISTICS(true);

  using Base = shm::InOrderLabelPropagation<LockingLpClusteringImpl, LockingLpClusteringConfig>;
  using AtomicClusterArray = scalable_vector<shm::parallel::IntegralAtomicWrapper<GlobalNodeID>>;

  using hasher_type = utils_tm::hash_tm::murmur2_hash;
  using allocator_type = growt::AlignedAllocator<>;
  // use 64 bit values for growt HT for now
  using table_type = typename growt::table_config<ClusterID, GlobalNodeWeight, hasher_type, allocator_type,
                                                  hmod::growable, hmod::deletion>::table_type;

  friend Base;
  friend Base::Base;

public:
  LockingLpClusteringImpl(const NodeID max_num_active_nodes, const NodeID max_num_nodes, const CoarseningContext &c_ctx)
      : Base{max_num_active_nodes, max_num_nodes},
        _c_ctx{c_ctx},
        _current_clustering(max_num_nodes),
        _next_clustering(max_num_nodes),
        _gain(max_num_active_nodes),
        _gain_buffer_index(max_num_active_nodes),
        _locked(max_num_active_nodes),
        _cluster_weights(max_num_nodes) {
    DBG << "Init HT with cap " << max_num_nodes;
    set_max_degree(c_ctx.lp.large_degree_threshold);
    set_max_num_neighbors(c_ctx.lp.max_num_neighbors);
  }

  const auto &compute_clustering(const DistributedGraph &graph, const NodeWeight max_cluster_weight) {
    initialize(&graph, graph.total_n()); // initializes _graph
    initialize_ghost_node_clusters();
    _max_cluster_weight = max_cluster_weight;
    DBG << V(_current_clustering) << V(_next_clustering);

    // catch special case where the coarse graph is larger than the fine graph due to an increased number of ghost nodes
    ensure_allocation_ok();

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
  //
  // Called from base class
  //
  //VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
  /*
   * Cluster weights
   *
   * Note: offset cluster IDs by 1 since growt cannot use 0 as key
   */

  auto create_handle_tmp() {
    DBG << "create handle";
    return _cluster_weights.get_handle();
  }

  auto &get_cluster_weights_handle() {
    thread_local static table_type::handle_type handle = create_handle_tmp();
    return handle;
  }

  void init_cluster_weight(const GlobalNodeID local_cluster, const NodeWeight weight) {
    const auto cluster = _graph->local_to_global_node(local_cluster);

    DBG << "insert(" << cluster + 1 << ", " << weight << ")";
    auto &handle = get_cluster_weights_handle(); // TODO
    [[maybe_unused]] const auto [it, success] = handle.insert(cluster + 1, weight);
    ASSERT(success);
  }

  NodeWeight cluster_weight(const GlobalNodeID cluster) {
    DBG << "find(" << cluster + 1 << ")";
    auto &handle = get_cluster_weights_handle(); // TODO
    auto it = handle.find(cluster + 1);
    ASSERT(it != handle.end()) << "Uninitialized cluster: " << cluster;

    return (*it).second;
  }

  void reset_node_state(const NodeID u) {
    Base::reset_node_state(u);
    _locked[u] = 0;
  }

  bool move_cluster_weight(const GlobalNodeID old_cluster, const GlobalNodeID new_cluster, const NodeWeight delta,
                           const NodeWeight max_weight) {
    if (cluster_weight(old_cluster) + delta <= max_weight) {
      auto &handle = get_cluster_weights_handle(); // TODO
      DBG << "update(" << old_cluster + 1 << ", ..., " << delta << ")";
      [[maybe_unused]] const auto [old_it, old_found] = handle.update(
          old_cluster + 1, [](auto &lhs, const auto rhs) { return lhs -= rhs; }, delta);
      ASSERT(old_it != handle.end() && old_found) << "Uninitialized cluster: " << old_cluster;

      DBG << "update(" << new_cluster + 1 << ", ..., " << delta << ")";
      [[maybe_unused]] const auto [new_it, new_found] = handle.update(
          new_cluster + 1, [](auto &lhs, const auto rhs) { return lhs += rhs; }, delta);
      ASSERT(new_it != handle.end() && new_found) << "Uninitialized cluster: " << new_cluster;

      DBG << "OK";
      return true;
    }
    DBG << "condition is false";
    return false;
  }

  void set_cluster_weight(const GlobalNodeID cluster, const NodeWeight weight) {
    auto &handle = get_cluster_weights_handle(); // TODO
    DBG << "insert_or_update(" << cluster + 1 << ", ..., " << weight << ")";
    handle.insert_or_update(
        cluster + 1, weight, [](auto &lhs, const auto rhs) { return lhs = rhs; }, weight);
  }

  void change_cluster_weight(const GlobalNodeID cluster, const NodeWeight delta) {
    DBG << "change_cluster_weight";

    auto &handle = get_cluster_weights_handle(); // TODO
    DBG << "update(" << cluster + 1 << ", ..., " << delta << ")";
    [[maybe_unused]] const auto [it, found] = handle.update(
        cluster + 1, [](auto &lhs, const auto rhs) { return lhs += rhs; }, delta);
    ASSERT(it != handle.end() && found) << "Uninitialized cluster: " << cluster;
  }

  [[nodiscard]] NodeWeight initial_cluster_weight(const NodeID u) const { return _graph->node_weight(u); }

  [[nodiscard]] NodeWeight max_cluster_weight(const GlobalNodeID /* cluster */) const { return _max_cluster_weight; }

  /*
   * Clusters
   */

  void init_cluster(const NodeID node, const NodeID cluster) {
    _current_clustering[node] = cluster;
    _next_clustering[node] = cluster;
  }

  [[nodiscard]] NodeID cluster(const NodeID u) const { return _next_clustering[u]; }

  void move_node(const NodeID node, const GlobalNodeID cluster) { _next_clustering[node] = cluster; }

  [[nodiscard]] GlobalNodeID initial_cluster(const NodeID u) const { return _graph->local_to_global_node(u); }

  /*
   * Moving nodes
   */

  [[nodiscard]] bool accept_cluster(const Base::ClusterSelectionState &state) const {
    return (state.current_gain > state.best_gain ||
            (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
           (state.current_cluster_weight + state.u_weight < max_cluster_weight(state.current_cluster) ||
            state.current_cluster == state.initial_cluster);
  }

  [[nodiscard]] inline bool activate_neighbor(const NodeID u) const { return !_locked[u] && _graph->is_owned_node(u); }
  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  //
  // Called from base class
  //
  //--------------------------------------------------------------------------------

private:
  void initialize_ghost_node_clusters() {
    tbb::parallel_for(_graph->n(), _graph->total_n(),
                      [&](const NodeID local_u) { init_cluster(local_u, _graph->local_to_global_node(local_u)); });
  }

  // a coarse graph could have a larger total size than the finer graph, since the number of ghost nodes could increase
  // arbitrarily -- thus, resize the rating map (only component depending on total_n()) in this special case
  // find a better solution to this issue in the future
  void ensure_allocation_ok() {
    SCOPED_TIMER("Allocation");

    if (_rating_map_ets.local().max_size() < _graph->total_n()) {
      for (auto &ets : _rating_map_ets) { ets.change_max_size(_graph->total_n()); }
    }
    if (_current_clustering.size() < _graph->total_n()) { _current_clustering.resize(_graph->total_n()); }
    if (_next_clustering.size() < _graph->total_n()) { _next_clustering.resize(_graph->total_n()); }
  }

  struct JoinRequest {
    GlobalNodeID global_requester;
    NodeWeight requester_weight;
    EdgeWeight requester_gain;
    GlobalNodeID global_requested;
  };

  struct JoinResponse {
    GlobalNodeID global_requester;
    NodeWeight new_weight;
    std::uint8_t response;
  };

  struct LabelMessage {
    GlobalNodeID global_node;
    GlobalNodeID global_new_label;
  };

  NodeID process_chunk(const NodeID from, const NodeID to) {
    const NodeID num_moved_nodes = perform_iteration(from, to);
    if (num_moved_nodes == 0) { return 0; } // nothing to do

    perform_distributed_moves(from, to);
    synchronize_labels(from, to);

    return num_moved_nodes;
  }

  void perform_distributed_moves(const NodeID from, const NodeID to) {
    // exchange join requests and collect them in _gain_buffer
    auto requests = mpi::graph::sparse_alltoall_interface_to_pe_get<JoinRequest>(
        *_graph, from, to,
        [&](const NodeID u) { return was_moved_during_round(u) && !_graph->contains_global_node(cluster(u)); },
        [&](const NodeID u) -> JoinRequest {
          DBG << "send join request: " << _graph->local_to_global_node(u) << " --> " << _next_clustering[u];
          return {.global_requester = _graph->local_to_global_node(u),
                  .requester_weight = _graph->node_weight(u),
                  .requester_gain = _gain[u],
                  .global_requested = _next_clustering[u]};
        });
    build_gain_buffer(requests);

    // perform moves
    tbb::parallel_for<NodeID>(0, _graph->n(), [&](const NodeID u) {
      const auto global_u = _graph->local_to_global_node(u);
      const auto to_cluster = cluster(global_u);

      for (std::size_t i = _gain_buffer_index[u]; i < _gain_buffer_index[u + 1]; ++i) {
        const auto [v, gain] = _gain_buffer[i];
        const auto v_weight = _graph->node_weight(v);
        const auto global_v = _graph->local_to_global_node(v);
        const auto from_cluster = cluster(global_v);

        if (move_cluster_weight(from_cluster, to_cluster, v_weight, max_cluster_weight(to_cluster))) {
          move_node(v, to_cluster);
          _locked[u] = 1;
        } else {
          break;
        }
      }
    });

    // build response messages
    std::vector<scalable_vector<JoinResponse>> responses;
    for (const auto &requests_from_pe : requests) { // allocate memory for responses
      responses.emplace_back(requests_from_pe.size());
    }

    std::vector<shm::parallel::IntegralAtomicWrapper<std::size_t>> next_message(requests.size());

    tbb::parallel_for<NodeID>(0, _graph->n(), [&](const NodeID u) {
      const auto global_u = _graph->local_to_global_node(u);
      const auto to_cluster = cluster(global_u);

      for (std::size_t i = _gain_buffer_index[u]; i < _gain_buffer_index[u + 1]; ++i) {
        const auto [v, gain] = _gain_buffer[i];
        const auto global_v = _graph->local_to_global_node(v);
        const auto pe = _graph->ghost_owner(v);
        const auto slot = next_message[pe]++;

        const std::uint8_t accepted = was_moved_during_round(v);
        responses[pe][slot] = JoinResponse{.global_requester = global_v,
                                           .new_weight = cluster_weight(to_cluster),
                                           .response = accepted};
      }
    });

    // exchange responses
    mpi::sparse_alltoall<JoinResponse, scalable_vector>(
        responses,
        [&](const auto buffer) {
          for (const auto [global_requester, new_weight, accepted] : buffer) {
            const auto local_requester = _graph->global_to_local_node(global_requester);
            set_cluster_weight(cluster(local_requester), new_weight);
            if (!accepted) { // if accepted, nothing to do, otherwise move back
              _next_clustering[local_requester] = _current_clustering[local_requester];
              change_cluster_weight(cluster(local_requester), _graph->node_weight(local_requester));
            }
          }
        },
        _graph->communicator());
  }

  void build_gain_buffer(auto &join_requests_per_pe) {
    // allocate memory only here since the number of ghost nodes could increase for coarse graphs
    TIMED_SCOPE("Allocation") {
      if (_gain_buffer.size() < _graph->ghost_n()) { _gain_buffer.resize(_graph->ghost_n()); }
    };

    // reset _gain_buffer_index
    _graph->pfor_nodes([&](const NodeID u) { _gain_buffer_index[u] = 0; });

    // build _gain_buffer_index and _gain_buffer arrays
    shm::parallel::parallel_for_over_chunks(join_requests_per_pe, [&](const JoinRequest &request) {
      _gain_buffer_index[_graph->global_to_local_node(request.global_requested)];
    });
    shm::parallel::prefix_sum(_gain_buffer_index.begin(), _gain_buffer_index.begin() + _graph->n() + 1,
                              _gain_buffer_index.begin());
    shm::parallel::parallel_for_over_chunks(join_requests_per_pe, [&](const JoinRequest &request) {
      const NodeID local_requested = _graph->global_to_local_node(request.global_requested);
      const NodeID local_requester = _graph->global_to_local_node(request.global_requester);
      _gain_buffer[--_gain_buffer_index[local_requested]] = {local_requester, request.requester_gain};
    });

    // sort buffer for each node by gain
    tbb::parallel_for<NodeID>(0, _graph->n(), [&](const NodeID u) {
      if (_gain_buffer_index[u] < _gain_buffer_index[u + 1]) {
        std::sort(_gain_buffer.begin() + _gain_buffer_index[u], _gain_buffer.begin() + _gain_buffer_index[u + 1],
                  [&](const auto &lhs, const auto &rhs) {
                    return lhs.second < rhs.second || (lhs.second == rhs.second && lhs.first < rhs.first);
                  });
      }
    });
  }

  //! Synchronize labels of ghost nodes.
  void synchronize_labels(const NodeID from, const NodeID to) {
    mpi::graph::sparse_alltoall_interface_to_pe<LabelMessage>(
        *_graph, from, to, [&](const NodeID u) { return was_moved_during_round(u); },
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

  [[nodiscard]] bool was_moved_during_round(const NodeID u) const {
    return _next_clustering[u] != _current_clustering[u];
  }

  using Base::_graph;

  const CoarseningContext &_c_ctx;

  NodeWeight _max_cluster_weight;
  AtomicClusterArray _current_clustering;
  AtomicClusterArray _next_clustering;
  scalable_vector<EdgeWeight> _gain;

  //! After receiving join requests, sort ghost nodes that want to join a cluster here. Use \c _gain_buffer_index to
  //! navigate this vector.
  scalable_vector<std::pair<NodeID, EdgeWeight>> _gain_buffer;
  //! After receiving join requests, sort ghost nodes that want to join a cluster into \c _gain_buffer. For each
  //! interface node, store the index for that nodes join requests in \c _gain_buffer in this vector.
  scalable_vector<shm::parallel::IntegralAtomicWrapper<NodeID>> _gain_buffer_index;

  scalable_vector<std::uint8_t> _locked;

  table_type _cluster_weights;
  //  tbb::enumerable_thread_specific<typename table_type::handle_type> _cluster_weights_handles_ets{
  //      [&] { return table_type::handle_type{_cluster_weights}; }};
};

LockingLpClustering::LockingLpClustering(const NodeID max_num_active_nodes, const NodeID max_num_nodes,
                                         const CoarseningContext &c_ctx)
    : _impl{std::make_unique<LockingLpClusteringImpl>(max_num_active_nodes, max_num_nodes, c_ctx)} {}

LockingLpClustering::~LockingLpClustering() = default;

const LockingLpClustering::AtomicClusterArray &
LockingLpClustering::compute_clustering(const DistributedGraph &graph, const NodeWeight max_cluster_weight) {
  return _impl->compute_clustering(graph, max_cluster_weight);
}
} // namespace dkaminpar