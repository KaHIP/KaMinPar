/*******************************************************************************
 * @file:   locking_lp_clustering.cc
 *
 * @author: Daniel Seemaier
 * @date:   01.10.21
 * @brief:
 ******************************************************************************/
#include "dkaminpar/coarsening/locking_label_propagation_clustering.h"

#include "dkaminpar/growt.h"
#include "dkaminpar/mpi_graph_utils.h"
#include "dkaminpar/utility/distributed_math.h"
#include "kaminpar/algorithm/parallel_label_propagation.h"

#include <unordered_set>

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
  SET_STATISTICS(true);

  using Base = shm::InOrderLabelPropagation<LockingLpClusteringImpl, LockingLpClusteringConfig>;
  using AtomicClusterArray = scalable_vector<shm::parallel::IntegralAtomicWrapper<GlobalNodeID>>;

  using hasher_type = utils_tm::hash_tm::murmur2_hash;
  using allocator_type = growt::AlignedAllocator<>;
  static_assert(std::numeric_limits<GlobalNodeWeight>::digits == 63 ||
                    std::numeric_limits<GlobalNodeWeight>::digits == 64,
                "use 64 bit value type"); // bug in growt with 32 bit values types (?)
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
        _gain_buffer_index(max_num_active_nodes + 1),
        _locked(max_num_active_nodes),
        _cluster_weights(max_num_nodes) {
    set_max_degree(c_ctx.lp.large_degree_threshold);
    set_max_num_neighbors(c_ctx.lp.max_num_neighbors);
  }

  const auto &compute_clustering(const DistributedGraph &graph, const NodeWeight max_cluster_weight) {
    initialize(&graph, graph.total_n()); // initializes _graph
    initialize_ghost_node_clusters();
    _max_cluster_weight = max_cluster_weight;

    // catch special case where the coarse graph is larger than the fine graph due to an increased number of ghost nodes
    ensure_allocation_ok();

    ASSERT(VALIDATE_INIT_STATE());

    const auto num_iterations = _c_ctx.lp.num_iterations == 0 ? std::numeric_limits<std::size_t>::max()
                                                              : _c_ctx.lp.num_iterations;

    for (std::size_t iteration = 0; iteration < num_iterations; ++iteration) {
      NodeID num_moved_nodes = 0;
      for (std::size_t chunk = 0; chunk < _c_ctx.lp.num_chunks; ++chunk) {
        const auto [from, to] = math::compute_local_range<NodeID>(_graph->n(), _c_ctx.lp.num_chunks, chunk);
        num_moved_nodes += process_chunk(from, to);
        mpi::barrier(_graph->communicator());
      }

      const GlobalNodeID num_moved_nodes_global = mpi::allreduce(static_cast<GlobalNodeID>(num_moved_nodes), MPI_SUM,
                                                                 _graph->communicator());
      if (num_moved_nodes_global == 0) { break; }
    }

    mpi::barrier(_graph->communicator());
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

  void init_cluster_weight(const GlobalNodeID local_cluster, const NodeWeight weight) {
    const auto cluster = _graph->local_to_global_node(local_cluster);

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

  void reset_node_state(const NodeID u) {
    Base::reset_node_state(u);
    _locked[u] = 0;
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

  bool move_cluster_weight_to(const GlobalNodeID cluster, const NodeWeight delta, const NodeWeight max_weight) {
    if (cluster_weight(cluster) + delta <= max_weight) {
      auto &handle = _cluster_weights_handles_ets.local();
      [[maybe_unused]] const auto [new_it, new_found] = handle.update(
          cluster + 1, [](auto &lhs, const auto rhs) { return lhs += rhs; }, delta);
      ASSERT(new_it != handle.end() && new_found) << "Uninitialized cluster: " << cluster + 1;
      return true;
    }
    return false;
  }

  void set_cluster_weight(const GlobalNodeID cluster, const NodeWeight weight) {
    auto &handle = _cluster_weights_handles_ets.local();
    handle.insert_or_update(
        cluster + 1, weight, [](auto &lhs, const auto rhs) { return lhs = rhs; }, weight);
  }

  void change_cluster_weight(const GlobalNodeID cluster, const NodeWeight delta) {
    auto &handle = _cluster_weights_handles_ets.local();
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
    ASSERT(node < _current_clustering.size() && node < _next_clustering.size());
    _current_clustering[node] = cluster;
    _next_clustering[node] = cluster;
  }

  [[nodiscard]] NodeID cluster(const NodeID u) const {
    ASSERT(u < _next_clustering.size());
    return _next_clustering[u];
  }

  void move_node(const NodeID node, const GlobalNodeID cluster) {
    _next_clustering[node] = cluster;
  }

  [[nodiscard]] GlobalNodeID initial_cluster(const NodeID u) const { return _graph->local_to_global_node(u); }

  /*
   * Moving nodes
   */

  [[nodiscard]] bool accept_cluster(const Base::ClusterSelectionState &state) {
    ASSERT(state.u < _locked.size() && !_locked[state.u]);

    SET_DEBUG(true);

    const bool ans = (state.current_gain > state.best_gain ||
                      (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
                     (state.current_cluster_weight + state.u_weight <= max_cluster_weight(state.current_cluster) ||
                      state.current_cluster == state.initial_cluster);
    if (ans) { _gain[state.u] = state.current_gain; }
    DBG << V(state.u) << V(state.current_cluster) << V(state.current_gain) << V(state.best_cluster)
        << V(state.best_gain) << V(ans) << V(state.current_cluster_weight) << V(state.u_weight)
        << V(max_cluster_weight(state.current_cluster));
    return ans;
  }

  [[nodiscard]] inline bool activate_neighbor(const NodeID u) const { return _graph->is_owned_node(u) && !_locked[u]; }
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
    GlobalNodeWeight cluster_weight;
    GlobalNodeID global_new_label;
  };

  NodeID process_chunk(const NodeID from, const NodeID to) {
    SET_DEBUG(false);

    if constexpr (kDebug) {
      mpi::barrier();
      LOG << "==============================";
      LOG << "process_chunk(" << from << ", " << to << ")";
      LOG << "==============================";
      mpi::barrier();
    }

    const NodeID num_moved_nodes = perform_iteration(from, to);
    SLOG << V(num_moved_nodes);

    // still has to take part in collective communication
    // if (num_moved_nodes == 0) { return 0; } // nothing to do

    if constexpr (kDebug) {
      for (const NodeID u : _graph->all_nodes()) {
        if (was_moved_during_round(u)) {
          const char prefix = _graph->is_owned_global_node(_next_clustering[u]) ? 'L' : 'G';
          DBG << u << ": " << _current_clustering[u] << " --> " << prefix << _next_clustering[u] << " G" << _gain[u]
              << " NW" << _graph->node_weight(u) << " NCW" << cluster_weight(_next_clustering[u]);
        }
      }
    }

    perform_distributed_moves(from, to);

    synchronize_labels(from, to);
    tbb::parallel_for<NodeID>(0, _graph->total_n(),
                              [&](const NodeID u) { _current_clustering[u] = _next_clustering[u]; });

    ASSERT(VALIDATE_STATE());

    return num_moved_nodes;
  }

  void perform_distributed_moves(const NodeID from, const NodeID to) {
    SET_DEBUG(true);

    mpi::barrier();
    LOG << "==============================";
    LOG << "perform_distributed_moves";
    LOG << "==============================";
    mpi::barrier();

    // exchange join requests and collect them in _gain_buffer
    auto requests = mpi::graph::sparse_alltoall_custom<JoinRequest>(
        *_graph, from, to,
        [&](const NodeID u) { return was_moved_during_round(u) && !_graph->is_owned_global_node(cluster(u)); },
        [&](const NodeID u) -> std::pair<JoinRequest, PEID> {
          const auto u_global = _graph->local_to_global_node(u);
          const auto u_weight = _graph->node_weight(u);
          const EdgeWeight u_gain = _gain[u];
          const GlobalNodeID new_cluster = _next_clustering[u];
          const PEID new_cluster_owner = _graph->find_owner_of_global_node(new_cluster);

          ASSERT(u_gain >= 0);

          DBG << "Join request: L" << u << "G" << _graph->local_to_global_node(u) << "={"
              << ".global_requester=" << _graph->local_to_global_node(u) << ", "
              << ".requester_weight=" << _graph->node_weight(u) << ", "
              << ".requester_gain=" << u_gain << ", "
              << ".global_requested=" << new_cluster << "} --> " << new_cluster_owner;

          return {{.global_requester = u_global,
                   .requester_weight = u_weight,
                   .requester_gain = u_gain,
                   .global_requested = new_cluster},
                  new_cluster_owner};
        });

    mpi::barrier();
    LOG << "==============================";
    LOG << "build_gain_buffer";
    LOG << "==============================";
    mpi::barrier();

    build_gain_buffer(requests);

    mpi::barrier();
    LOG << "==============================";
    LOG << "perform moves from gain buffer";
    LOG << "==============================";
    mpi::barrier();

    // allocate memory for response messages
    std::vector<scalable_vector<JoinResponse>> responses;
    for (const auto &requests_from_pe : requests) { // allocate memory for responses
      responses.emplace_back(requests_from_pe.size());
    }
    std::vector<shm::parallel::IntegralAtomicWrapper<std::size_t>> next_message(requests.size());

    // perform moves
    DBG << V(_gain_buffer_index);
    const PEID rank = mpi::get_comm_rank(_graph->communicator());

    tbb::parallel_for<NodeID>(0, _graph->n(), [&](const NodeID u) {
      // TODO stop trying to insert nodes after the first insert operation failed?

      const bool was_self_contained = _graph->local_to_global_node(u) == _current_clustering[u];
      const bool is_self_contained = _graph->local_to_global_node(u) == _next_clustering[u];
      const bool can_accept_nodes = was_self_contained && is_self_contained;

      for (std::size_t i = _gain_buffer_index[u]; i < _gain_buffer_index[u + 1]; ++i) {
        DBG << V(i) << V(u) << V(_gain_buffer[i].global_node) << V(_gain_buffer[i].node_weight)
            << V(_gain_buffer[i].gain);

        auto to_cluster = cluster(u);
        const auto [global_v, v_weight, gain] = _gain_buffer[i];
        const auto pe = _graph->find_owner_of_global_node(global_v);
        const auto slot = next_message[pe]++;

        bool accepted = can_accept_nodes;

        // we may accept a move anyways if OUR move request is symmetric, i.e., the leader of the cluster u wants to
        // join also wants to merge with u --> in this case, tie-break with no. of edges, if equal with lower owning PE
        if (!accepted && was_self_contained && _next_clustering[u] == global_v) {
          const EdgeID my_m = _graph->m();
          const EdgeID their_m = _graph->m(pe);
          accepted = (my_m < their_m) || ((my_m == their_m) && (rank < pe));

          _next_clustering[u] = _current_clustering[u];
          to_cluster = cluster(u); // use u's old cluster -- the one v knows
        } // TODO weight problem

        accepted = accepted && move_cluster_weight_to(to_cluster, v_weight, max_cluster_weight(to_cluster));
        if (accepted) {
          DBG << "Locking node " << u;
          _locked[u] = 1;
          _active[u] = 0;
        }

        // use weight entries to temporarily store u -- replace it by the correct weight in the next iteration
        static_assert(std::numeric_limits<NodeID>::digits == std::numeric_limits<NodeWeight>::digits + 1);
        NodeWeight u_as_weight;
        std::memcpy(&u_as_weight, &u, sizeof(NodeWeight));

        responses[pe][slot] = JoinResponse{.global_requester = global_v,
                                           .new_weight = u_as_weight,
                                           .response = accepted};

        DBG << "Response to -->" << global_v << ", label " << to_cluster << ": " << accepted;
      }
    });

    // second iteration to set weight in response messages
    shm::parallel::parallel_for_over_chunks(responses, [&](auto &entry) {
      NodeID u;
      std::memcpy(&u, &entry.new_weight, sizeof(NodeID));
      entry.new_weight = cluster_weight(cluster(u));
    });

    // exchange responses
    mpi::sparse_alltoall<JoinResponse, scalable_vector>(
        responses,
        [&](const auto buffer) {
          for (const auto [global_requester, new_weight, accepted] : buffer) {
            DBG << "Response for " << global_requester << ": " << (accepted ? 1 : 0) << ", " << new_weight;

            const auto local_requester = _graph->global_to_local_node(global_requester);
            ASSERT(!accepted || _locked[local_requester] == 0);

            // update weight of cluster that we want to join in any case
            set_cluster_weight(cluster(local_requester), new_weight);

            if (!accepted) { // if accepted, nothing to do, otherwise move back
              _next_clustering[local_requester] = _current_clustering[local_requester];
              change_cluster_weight(cluster(local_requester), _graph->node_weight(local_requester));

              // if required for symmetric matching, i.e., u -> v and v -> u matched
              if (!_locked[local_requester]) { _active[local_requester] = 1; }
            }
          }
        },
        _graph->communicator());
  }

  void build_gain_buffer(auto &join_requests_per_pe) {
    SET_DEBUG(true);

    mpi::barrier();
    LOG << "==============================";
    LOG << "build_gain_buffer";
    LOG << "==============================";
    mpi::barrier();

    ASSERT(_graph->n() <= _gain_buffer_index.size())
        << "_gain_buffer_index not large enough: " << _graph->n() << " > " << _gain_buffer_index.size();

    // reset _gain_buffer_index
    _graph->pfor_nodes([&](const NodeID u) { _gain_buffer_index[u] = 0; });
    _gain_buffer_index[_graph->n()] = 0;

    // build _gain_buffer_index and _gain_buffer arrays
    shm::parallel::parallel_for_over_chunks(join_requests_per_pe, [&](const JoinRequest &request) {
      const GlobalNodeID global_node = request.global_requested;
      ASSERT(_graph->is_owned_global_node(global_node));
      const NodeID local_node = _graph->global_to_local_node(global_node);
      ASSERT(local_node < _gain_buffer_index.size());

      ++_gain_buffer_index[local_node];
    });

    if constexpr (kDebug) {
      for (const NodeID u : _graph->nodes()) {
        if (_gain_buffer_index[u] > 0) { DBG << _gain_buffer_index[u] << " requests for " << u; }
      }
    }

    shm::parallel::prefix_sum(_gain_buffer_index.begin(), _gain_buffer_index.begin() + _graph->n() + 1,
                              _gain_buffer_index.begin());

    // allocate buffer
    TIMED_SCOPE("Allocation") { _gain_buffer.resize(_gain_buffer_index[_graph->n() - 1]); };

    shm::parallel::parallel_for_over_chunks(join_requests_per_pe, [&](const JoinRequest &request) {
      ASSERT(_graph->is_owned_global_node(request.global_requested));
      const NodeID local_requested = _graph->global_to_local_node(request.global_requested);
      ASSERT(local_requested < _gain_buffer_index.size());
      const NodeID global_requester = request.global_requester;

      ASSERT(_gain_buffer_index[local_requested] - 1 < _gain_buffer.size())
          << V(_gain_buffer_index[local_requested] - 1) << (_gain_buffer.size());
      _gain_buffer[--_gain_buffer_index[local_requested]] = {global_requester, request.requester_weight,
                                                             request.requester_gain};
    });

    if constexpr (kDebug) {
      std::ostringstream gain_buffer_str;
      gain_buffer_str << "UNSORTED\n";
      for (const auto &entry : _gain_buffer) {
        gain_buffer_str << "{.global_node=" << entry.global_node << ", .node_weight=" << entry.node_weight
                        << ", .gain=" << entry.gain << "} ";
      }
    }

    // sort buffer for each node by gain
    tbb::parallel_for<NodeID>(0, _graph->n(), [&](const NodeID u) {
      ASSERT(u + 1 < _gain_buffer_index.size());
      if (_gain_buffer_index[u] < _gain_buffer_index[u + 1]) {
        std::sort(_gain_buffer.begin() + _gain_buffer_index[u], _gain_buffer.begin() + _gain_buffer_index[u + 1],
                  [&](const auto &lhs, const auto &rhs) {
                    ASSERT(lhs.global_node != rhs.global_node);
                    const double lhs_rel_gain = 1.0 * lhs.gain / lhs.node_weight;
                    const double rhs_rel_gain = 1.0 * rhs.gain / rhs.node_weight;
                    return lhs_rel_gain < rhs_rel_gain ||
                           (lhs_rel_gain == rhs_rel_gain && lhs.global_node < rhs.global_node);
                  });
      }
    });

    if constexpr (kDebug) {
      std::ostringstream gain_buffer_str;
      gain_buffer_str << "SORTED\n";
      for (const auto &entry : _gain_buffer) {
        gain_buffer_str << "{.global_node=" << entry.global_node << ", .node_weight=" << entry.node_weight
                        << ", .gain=" << entry.gain << "} ";
      }
      SLOG << gain_buffer_str.str();
    }
  }

  //! Synchronize labels of ghost nodes.
  void synchronize_labels(const NodeID from, const NodeID to) {
    mpi::graph::sparse_alltoall_interface_to_pe<LabelMessage>(
        *_graph, from, to, [&](const NodeID u) { return was_moved_during_round(u); },
        [&](const NodeID u) -> LabelMessage {
          return {_graph->local_to_global_node(u), cluster_weight(cluster(u)), cluster(u)};
        },
        [&](const auto buffer) {
          tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
            const auto [global_node, cluster_weight, global_new_label] = buffer[i];
            const auto local_node = _graph->global_to_local_node(global_node);
            move_node(local_node, global_new_label);
            set_cluster_weight(global_new_label, cluster_weight);
          });
        });
  }

  [[nodiscard]] bool was_moved_during_round(const NodeID u) const {
    ASSERT(u < _next_clustering.size() && u < _current_clustering.size());
    return _next_clustering[u] != _current_clustering[u];
  }

#ifdef KAMINPAR_ENABLE_ASSERTIONS
  bool VALIDATE_INIT_STATE() {
    ASSERT(_graph->total_n() <= _current_clustering.size());
    ASSERT(_graph->total_n() <= _next_clustering.size());
    ASSERT(_graph->n() <= _locked.size());
    ASSERT(_graph->n() <= _gain.size());

    for (const NodeID u : _graph->all_nodes()) {
      ASSERT(_current_clustering[u] == _next_clustering[u]);
      ASSERT(cluster(u) == _graph->local_to_global_node(u));
      ASSERT(cluster_weight(cluster(u)) == _graph->node_weight(u));
    }

    return true;
  }

  bool VALIDATE_STATE() {
    for (const NodeID u : _graph->all_nodes()) { ASSERT(_current_clustering[u] == _next_clustering[u]); }
    ASSERT(VALIDATE_LOCKING_INVARIANT());
    return true;
  }

  bool VALIDATE_LOCKING_INVARIANT() {
    // set of nonempty labels on this PE
    std::unordered_set<GlobalNodeID> nonempty_labels;
    for (const NodeID u : _graph->nodes()) { nonempty_labels.insert(cluster(u)); }

    mpi::graph::sparse_alltoall_custom<GlobalNodeID>(
        *_graph, 0, _graph->n(),
        [&](const NodeID u) {
          ASSERT(cluster(u) < _graph->global_n());
          return !_graph->is_owned_global_node(cluster(u));
        },
        [&](const NodeID u) { return std::make_pair(cluster(u), _graph->find_owner_of_global_node(cluster(u))); },
        [&](const auto &buffer, const PEID pe) {
          for (const GlobalNodeID label : buffer) {
            ASSERT(nonempty_labels.contains(label))
                << label << " from PE " << pe << " does not exist on PE " << mpi::get_comm_rank(MPI_COMM_WORLD);
          }
        });

    return true;
  }
#endif // KAMINPAR_ENABLE_ASSERTIONS

  using Base::_graph;

  const CoarseningContext &_c_ctx;

  NodeWeight _max_cluster_weight{kInvalidNodeWeight};
  AtomicClusterArray _current_clustering;
  AtomicClusterArray _next_clustering;
  scalable_vector<EdgeWeight> _gain;

  struct GainBufferEntry {
    GlobalNodeID global_node;
    NodeWeight node_weight;
    EdgeWeight gain;
  };

  //! After receiving join requests, sort ghost nodes that want to join a cluster here. Use \c _gain_buffer_index to
  //! navigate this vector.
  scalable_vector<GainBufferEntry> _gain_buffer;
  //! After receiving join requests, sort ghost nodes that want to join a cluster into \c _gain_buffer. For each
  //! interface node, store the index for that nodes join requests in \c _gain_buffer in this vector.
  scalable_vector<shm::parallel::IntegralAtomicWrapper<NodeID>> _gain_buffer_index;

  scalable_vector<std::uint8_t> _locked;

  table_type _cluster_weights;
  tbb::enumerable_thread_specific<typename table_type::handle_type> _cluster_weights_handles_ets{
      [&] { return table_type::handle_type{_cluster_weights}; }};
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