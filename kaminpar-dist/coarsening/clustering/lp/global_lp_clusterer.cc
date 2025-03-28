/*******************************************************************************
 * Label propagation with clusters that can grow to multiple PEs.
 *
 * @file:   global_lp_clusterer.cc
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#include "kaminpar-dist/coarsening/clustering/lp/global_lp_clusterer.h"

#include "kaminpar-mpi/sparse_alltoall.h"

#include "kaminpar-dist/datastructures/distributed_graph.h"
#include "kaminpar-dist/datastructures/growt.h"
#include "kaminpar-dist/distributed_label_propagation.h"
#include "kaminpar-dist/graphutils/communication.h"
#include "kaminpar-dist/timer.h"

#include "kaminpar-common/datastructures/rating_map.h"

namespace kaminpar::dist {

namespace {

SET_DEBUG(false);

}

namespace {

struct GlobalLPClusteringConfig : public LabelPropagationConfig {
  using RatingMap = ::kaminpar::RatingMap<EdgeWeight, GlobalNodeID, rm_backyard::Sparsehash>;

  using ClusterID = GlobalNodeID;
  using ClusterWeight = GlobalNodeWeight;

  static constexpr bool kTrackClusterCount = false;   // NOLINT
  static constexpr bool kUseTwoHopClustering = false; // NOLINT
  static constexpr bool kUseActiveSetStrategy = true; // NOLINT
};

} // namespace

struct GlobalLPClusteringMemoryContext : public LabelPropagationMemoryContext<
                                             GlobalLPClusteringConfig::RatingMap,
                                             GlobalLPClusteringConfig::ClusterID> {
  StaticArray<GlobalNodeID> changed_label;
  StaticArray<std::uint8_t> locked;
  growt::GlobalNodeIDMap<GlobalNodeWeight> cluster_weights{0};
  StaticArray<GlobalNodeWeight> local_cluster_weights;
  growt::GlobalNodeIDMap<GlobalNodeWeight> weight_deltas{0};
};

template <typename Graph>
class GlobalLPClusteringImpl final : public ChunkRandomdLabelPropagation<
                                         GlobalLPClusteringImpl<Graph>,
                                         GlobalLPClusteringConfig,
                                         Graph>,
                                     public NonatomicClusterVectorRef<NodeID, GlobalNodeID> {
  using Base =
      ChunkRandomdLabelPropagation<GlobalLPClusteringImpl<Graph>, GlobalLPClusteringConfig, Graph>;
  using ClusterBase = NonatomicClusterVectorRef<NodeID, GlobalNodeID>;
  using WeightDeltaMap = growt::GlobalNodeIDMap<GlobalNodeWeight>;

  using Config = GlobalLPClusteringConfig;
  using ClusterID = Config::ClusterID;
  using ClusterWeight = Config::ClusterWeight;

public:
  explicit GlobalLPClusteringImpl(const Context &ctx)
      : _ctx(ctx),
        _c_ctx(ctx.coarsening),
        _passive_high_degree_threshold(_c_ctx.global_lp.passive_high_degree_threshold) {
    set_max_num_iterations(_c_ctx.global_lp.num_iterations);
    Base::set_max_degree(_c_ctx.global_lp.active_high_degree_threshold);
    Base::set_max_num_neighbors(_c_ctx.global_lp.max_num_neighbors);

    if (_c_ctx.global_lp.active_set_strategy == ActiveSetStrategy::LOCAL) {
      Base::enable_local_active_set();
    }
    if (_c_ctx.global_lp.active_set_strategy == ActiveSetStrategy::GLOBAL) {
      Base::enable_active_set();
    }
  }

  void setup(GlobalLPClusteringMemoryContext &memory_context) {
    Base::setup(memory_context);
    _changed_label = std::move(memory_context.changed_label);
    _locked = std::move(memory_context.locked);
    _cluster_weights = std::move(memory_context.cluster_weights);
    _local_cluster_weights = std::move(memory_context.local_cluster_weights);
    _weight_deltas = std::move(memory_context.weight_deltas);
  }

  GlobalLPClusteringMemoryContext release() {
    _weight_delta_handles_ets.clear();
    _cluster_weights_handles_ets.clear();

    auto [rating_map_ets, active, favored_clusters] = Base::release();
    return {
        {
            std::move(rating_map_ets),
            std::move(active),
            std::move(favored_clusters),
        },
        std::move(_changed_label),
        std::move(_locked),
        std::move(_cluster_weights),
        std::move(_local_cluster_weights),
        std::move(_weight_deltas),
    };
  }

  void preinitialize(const NodeID num_nodes, const NodeID num_active_nodes) {
    Base::preinitialize(num_nodes, num_active_nodes, num_nodes);
  }

  void initialize(const Graph &graph) {
    _graph = &graph;

    if (_c_ctx.global_lp.active_set_strategy == ActiveSetStrategy::GLOBAL) {
      // Dummy access to initialize the ghost graph
      _graph->ghost_graph();
    }

    START_TIMER("Initialize high-degree node info");
    if (_passive_high_degree_threshold > 0) {
      SCOPED_HEAP_PROFILER("Initialize high-degree node info");
      graph.init_high_degree_info(_passive_high_degree_threshold);
    }
    STOP_TIMER();

    TIMER_BARRIER(graph.communicator());
    START_TIMER("Allocation");
    allocate(graph);
    STOP_TIMER();

    TIMER_BARRIER(graph.communicator());
    START_TIMER("Initialize datastructures");
    START_HEAP_PROFILER("Initialize datastructures");
    _cluster_weights_handles_ets.clear();
    _cluster_weights = ClusterWeightsMap{0};
    std::fill(_local_cluster_weights.begin(), _local_cluster_weights.end(), 0);

    Base::initialize(&graph, graph.total_n());
    initialize_ghost_node_clusters();
    STOP_HEAP_PROFILER();
    STOP_TIMER();
  }

  void set_max_cluster_weight(const GlobalNodeWeight weight) {
    _max_cluster_weight = weight;
  }

  void compute_clustering(StaticArray<GlobalNodeID> &clustering, const Graph &graph) {
    TIMER_BARRIER(graph.communicator());
    SCOPED_TIMER("Label propagation");
    SCOPED_HEAP_PROFILER("Label Propagation");

    init_clusters_ref(clustering);
    initialize(graph);

    const int num_chunks = _c_ctx.global_lp.chunks.compute(_ctx.parallel);

    SCOPED_HEAP_PROFILER("Process chunks");
    for (int iteration = 0; iteration < _max_num_iterations; ++iteration) {
      GlobalNodeID global_num_moved_nodes = 0;
      for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const auto [from, to] = math::compute_local_range<NodeID>(_graph->n(), num_chunks, chunk);
        global_num_moved_nodes += process_chunk(from, to);
      }

      if constexpr (kDebug) {
        GlobalNodeID global_num_skipped_nodes = Base::_num_skipped_nodes_ets.combine(std::plus{});
        MPI_Allreduce(
            MPI_IN_PLACE,
            &global_num_skipped_nodes,
            1,
            mpi::type::get<GlobalNodeID>(),
            MPI_SUM,
            _graph->communicator()
        );

        DBG0 << "Iteration " << iteration << ": " << global_num_moved_nodes << " nodes moved, "
             << global_num_skipped_nodes << " nodes skipped";
        Base::_num_skipped_nodes_ets.clear();
      }

      if (global_num_moved_nodes == 0) {
        break;
      }
    }
  }

  void set_max_num_iterations(const int max_num_iterations) {
    _max_num_iterations =
        max_num_iterations == 0 ? std::numeric_limits<int>::max() : max_num_iterations;
  }

  //--------------------------------------------------------------------------------
  //
  // Called from base class
  //
  // VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
  void reset_node_state(const NodeID u) {
    Base::reset_node_state(u);
    _changed_label[u] = kInvalidGlobalNodeID;
  }

  /*
   * Cluster weights
   * Note: offset cluster IDs by 1 since growt cannot use 0 as key.
   */

  void init_cluster_weight(const ClusterID lcluster, const ClusterWeight weight) {
    if (_graph->is_owned_node(lcluster)) {
      __atomic_store_n(&_local_cluster_weights[lcluster], weight, __ATOMIC_RELAXED);
    } else {
      KASSERT(lcluster < _graph->total_n());
      const auto gcluster = _graph->local_to_global_node(static_cast<NodeID>(lcluster));
      auto &handle = _cluster_weights_handles_ets.local();
      [[maybe_unused]] const auto [it, success] = handle.insert(gcluster + 1, weight);
      KASSERT(success, "Cluster already initialized: " << gcluster + 1);
    }
  }

  ClusterWeight cluster_weight(const ClusterID gcluster) {
    if (_graph->is_owned_global_node(gcluster)) {
      const NodeID lcluster = _graph->global_to_local_node(gcluster);
      return __atomic_load_n(&_local_cluster_weights[lcluster], __ATOMIC_RELAXED);
    } else {
      auto &handle = _cluster_weights_handles_ets.local();
      auto it = handle.find(gcluster + 1);
      KASSERT(it != handle.end(), "read weight of uninitialized cluster: " << gcluster);
      return (*it).second;
    }
  }

  bool move_cluster_weight(
      const ClusterID old_gcluster,
      const ClusterID new_gcluster,
      const ClusterWeight weight_delta,
      const ClusterWeight max_weight,
      const bool check_weight_constraint = true
  ) {
    // Reject move if it violates local weight constraint
    if (check_weight_constraint && cluster_weight(new_gcluster) + weight_delta > max_weight) {
      return false;
    }

    auto &handle = _cluster_weights_handles_ets.local();

    if (_graph->is_owned_global_node(old_gcluster)) {
      const NodeID old_lcluster = _graph->global_to_local_node(old_gcluster);
      __atomic_fetch_sub(&_local_cluster_weights[old_lcluster], weight_delta, __ATOMIC_RELAXED);
    } else {
      // Otherwise, move node to new cluster
      [[maybe_unused]] const auto [it, found] = handle.update(
          old_gcluster + 1, [](auto &lhs, const auto rhs) { return lhs -= rhs; }, weight_delta
      );
      KASSERT(
          it != handle.end() && found, "moved weight from uninitialized cluster: " << old_gcluster
      );
    }

    if (_graph->is_owned_global_node(new_gcluster)) {
      const NodeID new_lcluster = _graph->global_to_local_node(new_gcluster);
      __atomic_fetch_add(&_local_cluster_weights[new_lcluster], weight_delta, __ATOMIC_RELAXED);
    } else {
      [[maybe_unused]] const auto [it, found] = handle.update(
          new_gcluster + 1, [](auto &lhs, const auto rhs) { return lhs += rhs; }, weight_delta
      );
      KASSERT(
          it != handle.end() && found, "moved weight to uninitialized cluster: " << new_gcluster
      );
    }

    return true;
  }

  void change_cluster_weight(
      const ClusterID gcluster, const ClusterWeight delta, [[maybe_unused]] const bool must_exist
  ) {
    if (_graph->is_owned_global_node(gcluster)) {
      const NodeID lcluster = _graph->global_to_local_node(gcluster);
      __atomic_fetch_add(&_local_cluster_weights[lcluster], delta, __ATOMIC_RELAXED);
    } else {
      auto &handle = _cluster_weights_handles_ets.local();

      [[maybe_unused]] const auto [it, not_found] = handle.insert_or_update(
          gcluster + 1, delta, [](auto &lhs, const auto rhs) { return lhs += rhs; }, delta
      );
      KASSERT(
          it != handle.end() && (!must_exist || !not_found),
          "changed weight of uninitialized cluster: " << gcluster
      );
    }
  }

  [[nodiscard]] NodeWeight initial_cluster_weight(const GlobalNodeID u) {
    KASSERT(u < _graph->total_n());
    return _graph->node_weight(static_cast<NodeID>(u));
  }

  [[nodiscard]] ClusterWeight max_cluster_weight(const GlobalNodeID /* cluster */) {
    return _max_cluster_weight;
  }

  /*
   * Clusters
   */

  void move_node(const NodeID lu, const ClusterID gcluster) {
    KASSERT(lu < _changed_label.size());
    _changed_label[lu] = this->cluster(lu);
    NonatomicClusterVectorRef::move_node(lu, gcluster);

    // Detect if a node was moved back to its original cluster
    if (_c_ctx.global_lp.prevent_cyclic_moves && gcluster == initial_cluster(lu)) {
      // If the node ID is the smallest among its non-local neighbors, lock the
      // node to its original cluster
      bool interface_node = false;
      bool smallest = true;

      _graph->adjacent_nodes(lu, [&](const NodeID lv) {
        if (_graph->is_owned_node(lv)) {
          return false;
        }

        interface_node = true;
        const GlobalNodeID gu = _graph->local_to_global_node(lu);
        const GlobalNodeID gv = _graph->local_to_global_node(lv);
        if (gv < gu) {
          smallest = false;
          return true;
        }

        return false;
      });

      if (interface_node && smallest) {
        _locked[lu] = 1;
      }
    }
  }

  [[nodiscard]] ClusterID initial_cluster(const NodeID u) {
    return _graph->local_to_global_node(u);
  }

  /*
   * Moving nodes
   */

  [[nodiscard]] bool accept_cluster(const Base::ClusterSelectionState &state) {
    return (state.current_gain > state.best_gain ||
            (state.current_gain == state.best_gain && state.local_rand.random_bool())) &&
           (state.current_cluster_weight + state.u_weight <=
                max_cluster_weight(state.current_cluster) ||
            state.current_cluster == state.initial_cluster);
  }

  [[nodiscard]] inline bool activate_neighbor(const NodeID u) {
    return _graph->is_owned_node(u);
  }

  [[nodiscard]] inline bool accept_neighbor(NodeID /* u */, const NodeID v) {
    return _passive_high_degree_threshold == 0 || !_graph->is_high_degree_node(v);
  }

  [[nodiscard]] inline bool skip_node(const NodeID lnode) {
    return _c_ctx.global_lp.prevent_cyclic_moves && _locked[lnode];
  }

  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  //
  // Called from base class
  //
  //--------------------------------------------------------------------------------

private:
  GlobalNodeID process_chunk(const NodeID from, const NodeID to) {
    TIMER_BARRIER(_graph->communicator());

    const NodeID local_num_moved_nodes = TIMED_SCOPE("Local work") {
      return Base::perform_iteration(from, to, kDebug);
    };

    const GlobalNodeID global_num_moved_nodes =
        mpi::allreduce(local_num_moved_nodes, MPI_SUM, _graph->communicator());

    control_cluster_weights(from, to);

    if (global_num_moved_nodes > 0) {
      synchronize_ghost_node_clusters(from, to);
    }

    if (_c_ctx.global_lp.merge_singleton_clusters) {
      cluster_isolated_nodes(from, to);
    }

    return global_num_moved_nodes;
  }

  void allocate(const Graph &graph) {
    SCOPED_HEAP_PROFILER("Allocation");

    if (_changed_label.size() < graph.n()) {
      _changed_label.resize(graph.n());
    }

    if (_local_cluster_weights.size() < graph.n()) {
      _local_cluster_weights.resize(graph.n());
    }

    Base::allocate();

    if (_c_ctx.global_lp.prevent_cyclic_moves) {
      _locked.resize(graph.n());
    }
  }

  void initialize_ghost_node_clusters() {
    tbb::parallel_for(_graph->n(), _graph->total_n(), [&](const NodeID local_u) {
      const GlobalNodeID label = _graph->local_to_global_node(local_u);
      init_cluster(local_u, label);
    });
  }

  void control_cluster_weights(const NodeID from, const NodeID to) {
    TIMER_BARRIER(_graph->communicator());
    SCOPED_TIMER("Synchronize cluster weights");

    if (!should_sync_cluster_weights()) {
      return;
    }

    const PEID size = mpi::get_comm_size(_graph->communicator());

    START_TIMER("Allocation");
    _weight_delta_handles_ets.clear();
    _weight_deltas = WeightDeltaMap(0);
    std::vector<parallel::Atomic<std::size_t>> num_messages(size);
    STOP_TIMER();

    TIMER_BARRIER(_graph->communicator());
    START_TIMER("Fill hash table");
    _graph->pfor_nodes(from, to, [&](const NodeID u) {
      if (_changed_label[u] != kInvalidGlobalNodeID) {
        auto &handle = _weight_delta_handles_ets.local();
        const GlobalNodeID old_label = _changed_label[u];
        const GlobalNodeID new_label = cluster(u);
        const NodeWeight weight = _graph->node_weight(u);

        if (!_graph->is_owned_global_node(old_label)) {
          auto [old_it, old_inserted] = handle.insert_or_update(
              old_label + 1, -weight, [&](auto &lhs, auto &rhs) { return lhs -= rhs; }, weight
          );
          if (old_inserted) {
            const PEID owner = _graph->find_owner_of_global_node(old_label);
            ++num_messages[owner];
          }
        }

        if (!_graph->is_owned_global_node(new_label)) {
          auto [new_it, new_inserted] = handle.insert_or_update(
              new_label + 1, weight, [&](auto &lhs, auto &rhs) { return lhs += rhs; }, weight
          );
          if (new_inserted) {
            const PEID owner = _graph->find_owner_of_global_node(new_label);
            ++num_messages[owner];
          }
        }
      }
    });
    STOP_TIMER();

    struct Message {
      GlobalNodeID cluster;
      GlobalNodeWeight delta;
    };

    TIMER_BARRIER(_graph->communicator());
    START_TIMER("Allocation");
    std::vector<NoinitVector<Message>> out_msgs(size);
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) { out_msgs[pe].resize(num_messages[pe]); });
    STOP_TIMER();

    TIMER_BARRIER(_graph->communicator());
    START_TIMER("Create messages");
    growt::pfor_handles(
        _weight_delta_handles_ets,
        [&](const GlobalNodeID gcluster_p1, const GlobalNodeWeight weight) {
          const GlobalNodeID gcluster = gcluster_p1 - 1;
          const PEID owner = _graph->find_owner_of_global_node(gcluster);
          const std::size_t index = num_messages[owner].fetch_sub(1) - 1;
          out_msgs[owner][index] = {.cluster = gcluster, .delta = weight};
        }
    );
    STOP_TIMER();

    TIMER_BARRIER(_graph->communicator());
    START_TIMER("Exchange messages");
    auto in_msgs = mpi::sparse_alltoall_get<Message>(out_msgs, _graph->communicator());
    STOP_TIMER();

    TIMER_BARRIER(_graph->communicator());
    START_TIMER("Integrate messages");
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
      tbb::parallel_for<std::size_t>(0, in_msgs[pe].size(), [&](const std::size_t i) {
        const auto [cluster, delta] = in_msgs[pe][i];
        change_cluster_weight(cluster, delta, false);
      });
    });

    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
      tbb::parallel_for<std::size_t>(0, in_msgs[pe].size(), [&](const std::size_t i) {
        const auto [cluster, delta] = in_msgs[pe][i];
        in_msgs[pe][i].delta = cluster_weight(cluster);
      });
    });
    STOP_TIMER();

    TIMER_BARRIER(_graph->communicator());
    START_TIMER("Exchange messages");
    auto in_resps = mpi::sparse_alltoall_get<Message>(in_msgs, _graph->communicator());
    STOP_TIMER();

    TIMER_BARRIER(_graph->communicator());
    START_TIMER("Integrate messages");
    parallel::Atomic<std::uint8_t> violation = 0;
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
      tbb::parallel_for<std::size_t>(0, in_resps[pe].size(), [&](const std::size_t i) {
        const auto [cluster, delta] = in_resps[pe][i];
        GlobalNodeWeight new_weight = delta;
        const GlobalNodeWeight old_weight = cluster_weight(cluster);

        if (delta > _max_cluster_weight) {
          const GlobalNodeWeight increase_by_others = new_weight - old_weight;

          auto &handle = _weight_delta_handles_ets.local();
          auto it = handle.find(cluster + 1);
          KASSERT(it != handle.end());
          const GlobalNodeWeight increase_by_me = (*it).second;

          violation = 1;
          new_weight =
              _max_cluster_weight + (1.0 * increase_by_me / (increase_by_others + increase_by_me)) *
                                        (new_weight - _max_cluster_weight);
        }
        change_cluster_weight(cluster, -old_weight + new_weight, true);
      });
    });
    STOP_TIMER();

    // Barrier has to be placed here since code paths might diverge after the return statement
    TIMER_BARRIER(_graph->communicator());

    // If we detected a max cluster weight violation, remove node weight
    // proportional to our chunk of the cluster weight
    if (!should_enforce_cluster_weights() || !violation) {
      return;
    }

    //
    // VVV possibly diverged code paths, might not be executed on all PEs VVV
    //

    START_TIMER("Enforce cluster weights");
    _graph->pfor_nodes(from, to, [&](const NodeID u) {
      const GlobalNodeID old_label = _changed_label[u];
      if (old_label == kInvalidGlobalNodeID) {
        return;
      }

      const GlobalNodeID new_label = cluster(u);
      const GlobalNodeWeight new_label_weight = cluster_weight(new_label);
      if (new_label_weight > _max_cluster_weight) {
        move_node(u, old_label);
        move_cluster_weight(new_label, old_label, _graph->node_weight(u), 0, false);
      }
    });
    STOP_TIMER();
  }

  void synchronize_ghost_node_clusters(const NodeID from, const NodeID to) {
    TIMER_BARRIER(_graph->communicator());
    SCOPED_TIMER("Synchronize ghost node clusters");

    struct ChangedLabelMessage {
      NodeID owner_lnode;
      ClusterID new_gcluster;
    };

    mpi::graph::sparse_alltoall_interface_to_pe<ChangedLabelMessage>(
        *_graph,
        from,
        to,
        [&](const NodeID lnode) { return _changed_label[lnode] != kInvalidGlobalNodeID; },
        [&](const NodeID lnode) -> ChangedLabelMessage { return {lnode, cluster(lnode)}; },
        [&](const auto &buffer, const PEID owner) {
          tbb::parallel_for(tbb::blocked_range<std::size_t>(0, buffer.size()), [&](const auto &r) {
            auto &weight_delta_handle = _weight_delta_handles_ets.local();

            for (std::size_t i = r.begin(); i != r.end(); ++i) {
              const auto [owner_lnode, new_gcluster] = buffer[i];

              const GlobalNodeID gnode = _graph->offset_n(owner) + owner_lnode;
              KASSERT(!_graph->is_owned_global_node(gnode));

              const NodeID lnode = _graph->global_to_local_node(gnode);
              const NodeWeight weight = _graph->node_weight(lnode);

              const GlobalNodeID old_gcluster = cluster(lnode);

              // If we synchronize the weights of clusters with local
              // changes, we already have the right weight including ghost
              // vertices --> only update weight if we did not get an update

              if (!should_sync_cluster_weights() ||
                  weight_delta_handle.find(old_gcluster + 1) == weight_delta_handle.end()) {
                change_cluster_weight(old_gcluster, -weight, true);
              }
              NonatomicClusterVectorRef::move_node(lnode, new_gcluster);
              if (!should_sync_cluster_weights() ||
                  weight_delta_handle.find(new_gcluster + 1) == weight_delta_handle.end()) {
                change_cluster_weight(new_gcluster, weight, false);
              }

              Base::activate_neighbors_of_ghost_node(lnode);
            }
          });
        }
    );

    _graph->pfor_nodes(from, to, [&](const NodeID lnode) {
      _changed_label[lnode] = kInvalidGlobalNodeID;
    });
  }

  /*!
   * Build clusters of isolated nodes: store the first isolated node and add
   * subsequent isolated nodes to its cluster until the maximum cluster weight
   * is violated; then, move on to the next isolated node etc.
   * @param from The first node to consider.
   * @param to One-after the last node to consider.
   */
  void cluster_isolated_nodes(const NodeID from, const NodeID to) {
    TIMER_BARRIER(_graph->communicator());
    SCOPED_TIMER("Cluster isolated nodes");

    tbb::enumerable_thread_specific<GlobalNodeID> isolated_node_ets(kInvalidNodeID);
    tbb::parallel_for(tbb::blocked_range<NodeID>(from, to), [&](tbb::blocked_range<NodeID> r) {
      NodeID current = isolated_node_ets.local();
      ClusterID current_cluster =
          current == kInvalidNodeID ? kInvalidGlobalNodeID : cluster(current);
      ClusterWeight current_weight =
          current == kInvalidNodeID ? kInvalidNodeWeight : cluster_weight(current_cluster);

      for (NodeID u = r.begin(); u != r.end(); ++u) {
        if (_graph->degree(u) == 0) {
          const auto u_cluster = cluster(u);
          const auto u_weight = cluster_weight(u_cluster);

          if (current != kInvalidNodeID &&
              current_weight + u_weight <= max_cluster_weight(u_cluster)) {
            change_cluster_weight(current_cluster, u_weight, true);
            NonatomicClusterVectorRef::move_node(u, current_cluster);
            current_weight += u_weight;
          } else {
            current = u;
            current_cluster = u_cluster;
            current_weight = u_weight;
          }
        }
      }

      isolated_node_ets.local() = current;
    });
  }

  [[nodiscard]] bool should_sync_cluster_weights() const {
    return _ctx.coarsening.global_lp.sync_cluster_weights;
  }

  [[nodiscard]] bool should_enforce_cluster_weights() const {
    return _ctx.coarsening.global_lp.enforce_cluster_weights;
  }

  using Base::_graph;
  const Context &_ctx;
  const CoarseningContext &_c_ctx;

  NodeWeight _max_cluster_weight = std::numeric_limits<NodeWeight>::max();
  int _max_num_iterations = std::numeric_limits<int>::max();

  // If a node was moved during the current iteration: its label before the move
  StaticArray<GlobalNodeID> _changed_label;

  // Used to lock nodes to prevent cyclic node moves
  StaticArray<std::uint8_t> _locked;

  // Weights of non-local clusters (i.e., cluster ID is owned by another PE)
  using ClusterWeightsMap = typename growt::GlobalNodeIDMap<GlobalNodeWeight>;
  ClusterWeightsMap _cluster_weights{0};
  tbb::enumerable_thread_specific<typename ClusterWeightsMap::handle_type>
      _cluster_weights_handles_ets{[&] {
        return _cluster_weights.get_handle();
      }};

  // Weights of local clusters (i.e., cluster ID is owned by this PE)
  StaticArray<GlobalNodeWeight> _local_cluster_weights;

  // Skip neighbors if their degree is larger than this threshold, never skip
  // neighbors if set to 0
  EdgeID _passive_high_degree_threshold = 0;

  WeightDeltaMap _weight_deltas{0};
  tbb::enumerable_thread_specific<WeightDeltaMap::handle_type> _weight_delta_handles_ets{[this] {
    return _weight_deltas.get_handle();
  }};
};

class GlobalLPClusteringImplWrapper {
public:
  GlobalLPClusteringImplWrapper(const Context &ctx)
      : _csr_impl(std::make_unique<GlobalLPClusteringImpl<DistributedCSRGraph>>(ctx)),
        _compressed_impl(std::make_unique<GlobalLPClusteringImpl<DistributedCompressedGraph>>(ctx)
        ) {}

  void set_max_cluster_weight(const GlobalNodeWeight weight) {
    _csr_impl->set_max_cluster_weight(weight);
    _compressed_impl->set_max_cluster_weight(weight);
  }

  void compute_clustering(StaticArray<GlobalNodeID> &clustering, const DistributedGraph &graph) {
    const auto compute_clustering = [&](auto &impl, const auto &graph) {
      impl.setup(_memory_context);
      impl.compute_clustering(clustering, graph);
      _memory_context = impl.release();
    };

    const NodeID num_nodes = graph.total_n();
    const NodeID num_active_nodes = graph.n();
    _csr_impl->preinitialize(num_nodes, num_active_nodes);
    _compressed_impl->preinitialize(num_nodes, num_active_nodes);

    graph.reified(
        [&](const DistributedCSRGraph &csr_graph) {
          GlobalLPClusteringImpl<DistributedCSRGraph> &impl = *_csr_impl;
          compute_clustering(impl, csr_graph);
        },
        [&](const DistributedCompressedGraph &compressed_graph) {
          GlobalLPClusteringImpl<DistributedCompressedGraph> &impl = *_compressed_impl;
          compute_clustering(impl, compressed_graph);
        }
    );
  }

private:
  GlobalLPClusteringMemoryContext _memory_context;
  std::unique_ptr<GlobalLPClusteringImpl<DistributedCSRGraph>> _csr_impl;
  std::unique_ptr<GlobalLPClusteringImpl<DistributedCompressedGraph>> _compressed_impl;
};

//
// Public interface
//

GlobalLPClusterer::GlobalLPClusterer(const Context &ctx)
    : _impl(std::make_unique<GlobalLPClusteringImplWrapper>(ctx)) {}

GlobalLPClusterer::~GlobalLPClusterer() = default;

void GlobalLPClusterer::set_max_cluster_weight(const GlobalNodeWeight weight) {
  _impl->set_max_cluster_weight(weight);
}

void GlobalLPClusterer::cluster(
    StaticArray<GlobalNodeID> &clustering, const DistributedGraph &graph
) {
  _impl->compute_clustering(clustering, graph);
}

} // namespace kaminpar::dist
