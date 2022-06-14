/*******************************************************************************
 * @file:   locking_lp_clustering.cc
 *
 * @author: Daniel Seemaier
 * @date:   01.10.21
 * @brief:
 ******************************************************************************/
#include "dkaminpar/coarsening/locking_label_propagation_clustering.h"

#include <unordered_map>
#include <unordered_set>

#include "dkaminpar/growt.h"
#include "dkaminpar/mpi/graph_communication.h"
#include "dkaminpar/utils/math.h"
#include "kaminpar/label_propagation.h"
#include "kaminpar/parallel/atomic.h"
#include "kaminpar/parallel/loops.h"

namespace dkaminpar {
namespace {
/*!
 * Large rating map based on a \c unordered_map. We need this since cluster IDs are global node IDs.
 */
struct UnorderedRatingMap {
    EdgeWeight& operator[](const GlobalNodeID key) {
        return map[key];
    }
    [[nodiscard]] auto& entries() {
        return map;
    }
    void clear() {
        map.clear();
    }
    std::size_t capacity() const {
        return std::numeric_limits<std::size_t>::max();
    }
    void                                         resize(const std::size_t /* capacity */) {}
    std::unordered_map<GlobalNodeID, EdgeWeight> map{};
};

struct LockingLabelPropagationClusteringConfig : shm::LabelPropagationConfig {
    using Graph         = DistributedGraph;
    using RatingMap     = ::kaminpar::RatingMap<EdgeWeight, UnorderedRatingMap>;
    using ClusterID     = GlobalNodeID;
    using ClusterWeight = NodeWeight;
};
} // namespace

class LockingLabelPropagationClusteringImpl
    : public shm::InOrderLabelPropagation<
          LockingLabelPropagationClusteringImpl, LockingLabelPropagationClusteringConfig> {
    SET_STATISTICS(true);

    using Base =
        shm::InOrderLabelPropagation<LockingLabelPropagationClusteringImpl, LockingLabelPropagationClusteringConfig>;
    using AtomicClusterArray = scalable_vector<shm::parallel::Atomic<GlobalNodeID>>;

    friend Base;
    friend Base::Base;

    struct Statistics {
        shm::parallel::Atomic<int>        num_move_accepted{0};
        shm::parallel::Atomic<int>        num_move_rejected{0};
        shm::parallel::Atomic<int>        num_moves{0};
        shm::parallel::Atomic<EdgeWeight> gain_accepted{0};
        shm::parallel::Atomic<EdgeWeight> gain_rejected{0};

        void print() const {
            LOG << shm::logger::CYAN << "LockingLabelPropagationClustering statistics:";
            LOG << shm::logger::CYAN
                << "- num_move_accepted: " << mpi::gather_statistics_str(num_move_accepted, MPI_COMM_WORLD);
            LOG << shm::logger::CYAN
                << "- num_move_rejected: " << mpi::gather_statistics_str(num_move_rejected, MPI_COMM_WORLD);
            LOG << shm::logger::CYAN << "- num_moves: " << mpi::gather_statistics_str(num_moves, MPI_COMM_WORLD);
            LOG << shm::logger::CYAN
                << "- gain_accepted: " << mpi::gather_statistics_str(gain_accepted, MPI_COMM_WORLD);
            LOG << shm::logger::CYAN
                << "- gain_rejected: " << mpi::gather_statistics_str(gain_rejected, MPI_COMM_WORLD);
        }

        void reset() {
            num_move_accepted = 0;
            num_move_rejected = 0;
            num_moves         = 0;
            gain_accepted     = 0;
            gain_rejected     = 0;
        }
    };

public:
    explicit LockingLabelPropagationClusteringImpl(const Context& ctx)
        : _c_ctx{ctx.coarsening},
          _cluster_weights{ctx.partition.local_n()} {
        set_max_degree(_c_ctx.global_lp.large_degree_threshold);
        set_max_num_neighbors(_c_ctx.global_lp.max_num_neighbors);
    }

    const auto& compute_clustering(const DistributedGraph& graph, const GlobalNodeWeight max_cluster_weight) {
        TIMED_SCOPE("Allocation") {
            allocate(graph);
        };
        SCOPED_TIMER("Locking Label Propagation");

        initialize(&graph, graph.total_n()); // initializes _graph
        initialize_ghost_node_clusters();
        _max_cluster_weight = max_cluster_weight;

#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
        KASSERT(VALIDATE_INIT_STATE(), "", assert::heavy);
#endif

        for (std::size_t iteration = 0; iteration < _c_ctx.global_lp.num_iterations; ++iteration) {
            NodeID num_moved_nodes = 0;
            for (std::size_t chunk = 0; chunk < _c_ctx.global_lp.num_chunks; ++chunk) {
                const auto [from, to] =
                    math::compute_local_range<NodeID>(_graph->n(), _c_ctx.global_lp.num_chunks, chunk);
                num_moved_nodes += process_chunk(from, to);
            }

            const GlobalNodeID num_moved_nodes_global =
                mpi::allreduce(static_cast<GlobalNodeID>(num_moved_nodes), MPI_SUM, _graph->communicator());
            if (num_moved_nodes_global == 0) {
                break;
            }
        }

        return _current_clustering;
    }

    void print_statistics() {
        if constexpr (!kStatistics) {
            return;
        }
    }

protected:
    void allocate(const DistributedGraph& graph) {
        const NodeID allocated_num_nodes        = _current_clustering.size();
        const NodeID allocated_num_active_nodes = _gain.size();

        if (allocated_num_nodes < graph.total_n()) {
            _current_clustering.resize(graph.total_n());
            _next_clustering.resize(graph.total_n());
        }

        if (allocated_num_active_nodes < graph.n()) {
            _gain.resize(graph.n());
            _gain_buffer_index.resize(graph.n() + 1);
            _locked.resize(graph.n());
        }

        Base::allocate(graph.total_n(), graph.n());
    }

    //--------------------------------------------------------------------------------
    //
    // Called from base class
    //
    // VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    /*
     * Cluster weights
     *
     * Note: offset cluster IDs by 1 since growt cannot use 0 as key
     */

    void init_cluster_weight(const GlobalNodeID local_cluster, const NodeWeight weight) {
        const auto cluster = _graph->local_to_global_node(local_cluster);

        auto& handle                              = _cluster_weights_handles_ets.local();
        [[maybe_unused]] const auto [it, success] = handle.insert(cluster + 1, weight);
        KASSERT(success);
    }

    NodeWeight cluster_weight(const GlobalNodeID cluster) {
        auto& handle = _cluster_weights_handles_ets.local();
        auto  it     = handle.find(cluster + 1);
        KASSERT(it != handle.end(), "Uninitialized cluster: " << cluster + 1);

        return (*it).second;
    }

    void reset_node_state(const NodeID u) {
        Base::reset_node_state(u);
        _locked[u] = 0;
    }

    bool move_cluster_weight(
        const GlobalNodeID old_cluster, const GlobalNodeID new_cluster, const NodeWeight delta,
        const NodeWeight max_weight) {
        if (cluster_weight(new_cluster) + delta <= max_weight) {
            auto& handle                                    = _cluster_weights_handles_ets.local();
            [[maybe_unused]] const auto [old_it, old_found] = handle.update(
                old_cluster + 1, [](auto& lhs, const auto rhs) { return lhs -= rhs; }, delta);
            KASSERT((old_it != handle.end() && old_found), "Uninitialized cluster: " << old_cluster + 1);

            [[maybe_unused]] const auto [new_it, new_found] = handle.update(
                new_cluster + 1, [](auto& lhs, const auto rhs) { return lhs += rhs; }, delta);
            KASSERT((new_it != handle.end() && new_found), "Uninitialized cluster: " << new_cluster + 1);

            return true;
        }
        return false;
    }

    bool move_cluster_weight_to(const GlobalNodeID cluster, const NodeWeight delta, const NodeWeight max_weight) {
        if (cluster_weight(cluster) + delta <= max_weight) {
            auto& handle                                    = _cluster_weights_handles_ets.local();
            [[maybe_unused]] const auto [new_it, new_found] = handle.update(
                cluster + 1, [](auto& lhs, const auto rhs) { return lhs += rhs; }, delta);
            KASSERT((new_it != handle.end() && new_found), "Uninitialized cluster: " << cluster + 1);
            return true;
        }
        return false;
    }

    void set_cluster_weight(const GlobalNodeID cluster, const NodeWeight weight) {
        auto& handle = _cluster_weights_handles_ets.local();
        handle.insert_or_update(
            cluster + 1, weight, [](auto& lhs, const auto rhs) { return lhs = rhs; }, weight);
    }

    void change_cluster_weight(const GlobalNodeID cluster, const NodeWeight delta) {
        auto& handle                            = _cluster_weights_handles_ets.local();
        [[maybe_unused]] const auto [it, found] = handle.update(
            cluster + 1, [](auto& lhs, const auto rhs) { return lhs += rhs; }, delta);
        KASSERT((it != handle.end() && found), "Uninitialized cluster: " << cluster);
    }

    [[nodiscard]] NodeWeight initial_cluster_weight(const NodeID u) {
        return _graph->node_weight(u);
    }

    [[nodiscard]] NodeWeight max_cluster_weight(const GlobalNodeID /* cluster */) {
        return _max_cluster_weight;
    }

    /*
     * Clusters
     */

    void init_cluster(const NodeID node, const GlobalNodeID cluster) {
        KASSERT((node < _current_clustering.size() && node < _next_clustering.size()));
        _current_clustering[node] = cluster;
        _next_clustering[node]    = cluster;
    }

    [[nodiscard]] NodeID cluster(const NodeID u) {
        KASSERT(u < _next_clustering.size());
        return _next_clustering[u];
    }

    void move_node(const NodeID node, const GlobalNodeID cluster) {
        _next_clustering[node] = cluster;
    }

    [[nodiscard]] GlobalNodeID initial_cluster(const NodeID u) {
        return _graph->local_to_global_node(u);
    }

    /*
     * Moving nodes
     */

    [[nodiscard]] bool accept_cluster(const Base::ClusterSelectionState& state) {
        KASSERT((state.u < _locked.size() && !_locked[state.u]));
        const bool ans = (state.current_gain > state.best_gain
                          || (state.current_gain == state.best_gain && state.local_rand.random_bool()))
                         && (state.current_cluster_weight + state.u_weight <= max_cluster_weight(state.current_cluster)
                             || state.current_cluster == state.initial_cluster);

        if (ans) {
            _gain[state.u] = state.current_gain;
        }

        SET_DEBUG(false);
        DBG << V(state.u) << V(state.current_cluster) << V(state.current_gain) << V(state.best_cluster)
            << V(state.best_gain) << V(ans) << V(state.current_cluster_weight) << V(state.u_weight)
            << V(max_cluster_weight(state.current_cluster));
        return ans;
    }

    [[nodiscard]] inline bool activate_neighbor(const NodeID u) {
        return _graph->is_owned_node(u) && !_locked[u];
    }
    //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //
    // Called from base class
    //
    //--------------------------------------------------------------------------------

private:
    void initialize_ghost_node_clusters() {
        tbb::parallel_for(_graph->n(), _graph->total_n(), [&](const NodeID local_u) {
            init_cluster(local_u, _graph->local_to_global_node(local_u));
        });
    }

    struct JoinRequest {
        GlobalNodeID global_requester;
        NodeWeight   requester_weight;
        EdgeWeight   requester_gain;
        GlobalNodeID global_requested;
    };

    struct JoinResponse {
        GlobalNodeID global_requester;
        NodeWeight   new_weight;
        std::uint8_t response;
    };

    struct LabelMessage {
        GlobalNodeID     global_node;
        GlobalNodeWeight cluster_weight;
        GlobalNodeID     global_new_label;
    };

    NodeID process_chunk(const NodeID from, const NodeID to) {
        SET_DEBUG(false);

        if constexpr (kDebug) {
            mpi::barrier(_graph->communicator());
            LOG << "==============================";
            LOG << "process_chunk(" << from << ", " << to << ")";
            LOG << "==============================";
            mpi::barrier(_graph->communicator());
        }

        const NodeID num_moved_nodes = TIMED_SCOPE("Label Propagation", TIMER_FINE) {
            return perform_iteration(from, to);
        };

        // still has to take part in collective communication
        // if (num_moved_nodes == 0) { return 0; } // nothing to do

        if constexpr (kDebug) {
            for (const NodeID u: _graph->all_nodes()) {
                if (was_moved_during_round(u)) {
                    const char prefix = _graph->is_owned_global_node(_next_clustering[u]) ? 'L' : 'G';
                    DBG << u << ": " << _current_clustering[u] << " --> " << prefix << _next_clustering[u] << " G"
                        << _gain[u] << " NW" << _graph->node_weight(u) << " NCW" << cluster_weight(_next_clustering[u]);
                }
            }
        }

        perform_distributed_moves(from, to);
        synchronize_labels(from, to);

        tbb::parallel_for<NodeID>(
            0, _graph->total_n(), [&](const NodeID u) { _current_clustering[u] = _next_clustering[u]; });

#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
        KASSERT(VALIDATE_STATE(), "", assert::heavy);
#endif

        return num_moved_nodes;
    }

    void perform_distributed_moves(const NodeID from, const NodeID to) {
        SET_DEBUG(false);
        SCOPED_TIMER("Distributed moves", TIMER_FINE);

        if constexpr (kDebug) {
            mpi::barrier(_graph->communicator());
            LOG << "==============================";
            LOG << "perform_distributed_moves";
            LOG << "==============================";
            mpi::barrier(_graph->communicator());
        }

        START_TIMER("Exchange join requests", TIMER_FINE);
        // exchange join requests and collect them in _gain_buffer
        auto requests = mpi::graph::sparse_alltoall_custom<JoinRequest>(
            *_graph, from, to,
            [&](const NodeID u) { return was_moved_during_round(u) && !_graph->is_owned_global_node(cluster(u)); },
            [&](const NodeID u) { return _graph->find_owner_of_global_node(_next_clustering[u]); },
            [&](const NodeID u) -> JoinRequest {
                const auto         u_global    = _graph->local_to_global_node(u);
                const auto         u_weight    = _graph->node_weight(u);
                const EdgeWeight   u_gain      = _gain[u];
                const GlobalNodeID new_cluster = _next_clustering[u];
                KASSERT(u_gain >= 0);

                DBG << "Join request: L" << u << "G" << _graph->local_to_global_node(u) << "={"
                    << ".global_requester=" << _graph->local_to_global_node(u) << ", "
                    << ".requester_weight=" << _graph->node_weight(u) << ", "
                    << ".requester_gain=" << u_gain << ", "
                    << ".global_requested=" << new_cluster << "} --> "
                    << _graph->find_owner_of_global_node(new_cluster);

                return {
                    .global_requester = u_global,
                    .requester_weight = u_weight,
                    .requester_gain   = u_gain,
                    .global_requested = new_cluster};
            });
        STOP_TIMER(TIMER_FINE);

        if constexpr (kDebug) {
            mpi::barrier(_graph->communicator());
            LOG << "==============================";
            LOG << "build_gain_buffer";
            LOG << "==============================";
            mpi::barrier(_graph->communicator());
        }

        build_gain_buffer(requests);

        if constexpr (kDebug) {
            mpi::barrier(_graph->communicator());
            LOG << "==============================";
            LOG << "perform moves from gain buffer";
            LOG << "==============================";
            mpi::barrier(_graph->communicator());
        }

        // allocate memory for response messages
        START_TIMER("Allocation", TIMER_FINE);
        std::vector<scalable_vector<JoinResponse>> responses;
        for (const auto& requests_from_pe: requests) { // allocate memory for responses
            responses.emplace_back(requests_from_pe.size());
        }
        std::vector<shm::parallel::Atomic<std::size_t>> next_message(requests.size());
        STOP_TIMER(TIMER_FINE);

        // perform moves
        DBG << V(_gain_buffer_index);
        const PEID rank = mpi::get_comm_rank(_graph->communicator());

        START_TIMER("Perform moves and create responses", TIMER_FINE);
        tbb::parallel_for<NodeID>(0, _graph->n(), [&](const NodeID u) {
            // TODO stop trying to insert nodes after the first insert operation failed?

            const bool was_self_contained = _graph->local_to_global_node(u) == _current_clustering[u];
            const bool is_self_contained  = _graph->local_to_global_node(u) == _next_clustering[u];
            const bool can_accept_nodes   = was_self_contained && is_self_contained;

            for (std::size_t i = _gain_buffer_index[u]; i < _gain_buffer_index[u + 1]; ++i) {
                DBG << V(i) << V(u) << V(_gain_buffer[i].global_node) << V(_gain_buffer[i].node_weight)
                    << V(_gain_buffer[i].gain);

                auto to_cluster                       = cluster(u);
                const auto [global_v, v_weight, gain] = _gain_buffer[i];
                const auto pe                         = _graph->find_owner_of_global_node(global_v);
                const auto slot                       = next_message[pe]++;

                bool accepted = can_accept_nodes;

                // we may accept a move anyways if OUR move request is symmetric, i.e., the leader of the cluster u
                // wants to join also wants to merge with u --> in this case, tie-break with no. of edges, if equal with
                // lower owning PE
                if (!accepted && was_self_contained && _next_clustering[u] == global_v) {
                    const EdgeID my_m    = _graph->m();
                    const EdgeID their_m = _graph->m(pe);
                    accepted             = (my_m < their_m) || ((my_m == their_m) && (rank < pe));

                    if (accepted) {
                        _next_clustering[u] = _current_clustering[u];
                        to_cluster          = _current_clustering[u]; // use u's old cluster -- the one v knows
                    }
                } // TODO weight problem

                accepted = accepted && move_cluster_weight_to(to_cluster, v_weight, max_cluster_weight(to_cluster));
                if (accepted) {
                    DBG << "Locking node " << u;
                    _locked[u] = 1;
                    _active[u] = 0;
                }

                // use weight entries to temporarily store u -- replace it by the correct weight in the next iteration
                KASSERT(std::numeric_limits<NodeID>::digits == std::numeric_limits<NodeWeight>::digits + 1); // TODO
                NodeWeight u_as_weight;
                std::memcpy(&u_as_weight, &u, sizeof(NodeWeight));

                responses[pe][slot] =
                    JoinResponse{.global_requester = global_v, .new_weight = u_as_weight, .response = accepted};

                DBG << "Response to -->" << global_v << ", label " << to_cluster << ": " << accepted;
            }
        });

        // second iteration to set weight in response messages
        shm::parallel::chunked_for(responses, [&](auto& entry) {
            NodeID u;
            std::memcpy(&u, &entry.new_weight, sizeof(NodeID));
            entry.new_weight = cluster_weight(cluster(u));
        });
        STOP_TIMER(TIMER_FINE);

        START_TIMER("Exchange join responses", TIMER_FINE);
        // exchange responses
        mpi::sparse_alltoall<JoinResponse>(
            std::move(responses),
            [&](const auto buffer) {
                for (const auto [global_requester, new_weight, accepted]: buffer) {
                    const auto local_requester = _graph->global_to_local_node(global_requester);
                    DBG << "Response for " << global_requester << ": " << (accepted ? 1 : 0) << ", " << new_weight
                        << " " << _current_clustering[local_requester] << " --> " << _next_clustering[local_requester];
                    KASSERT((!accepted || _locked[local_requester] == 0));

                    // update weight of cluster that we want to join in any case
                    set_cluster_weight(cluster(local_requester), new_weight);

                    if (!accepted) { // if accepted, nothing to do, otherwise move back
                        _next_clustering[local_requester] = _current_clustering[local_requester];
                        change_cluster_weight(cluster(local_requester), _graph->node_weight(local_requester));

                        // if required for symmetric matching, i.e., u -> v and v -> u matched
                        if (!_locked[local_requester]) {
                            _active[local_requester] = 1;
                        }
                    }
                }
            },
            _graph->communicator());
        STOP_TIMER(TIMER_FINE);
    }

    template <typename JoinRequests>
    void build_gain_buffer(JoinRequests& join_requests_per_pe) {
        SET_DEBUG(false);
        SCOPED_TIMER("Build gain buffer", TIMER_FINE);

        if constexpr (kDebug) {
            mpi::barrier(_graph->communicator());
            LOG << "==============================";
            LOG << "build_gain_buffer";
            LOG << "==============================";
            mpi::barrier(_graph->communicator());
        }

        KASSERT(
            _graph->n() <= _gain_buffer_index.size(),
            "_gain_buffer_index not large enough: " << _graph->n() << " > " << _gain_buffer_index.size());

        // reset _gain_buffer_index
        TIMED_SCOPE("Reset gain buffer", TIMER_FINE) {
            _graph->pfor_nodes([&](const NodeID u) { _gain_buffer_index[u] = 0; });
            _gain_buffer_index[_graph->n()] = 0;
        };

        // build _gain_buffer_index and _gain_buffer arrays
        START_TIMER("Build index buffer", TIMER_FINE);

        shm::parallel::chunked_for(join_requests_per_pe, [&](const JoinRequest& request) {
            const GlobalNodeID global_node = request.global_requested;
            const NodeID       local_node  = _graph->global_to_local_node(global_node);
            ++_gain_buffer_index[local_node];
        });

        START_TIMER("Prefix sum", TIMER_FINE);
        shm::parallel::prefix_sum(
            _gain_buffer_index.begin(), _gain_buffer_index.begin() + _graph->n() + 1, _gain_buffer_index.begin());
        STOP_TIMER(TIMER_FINE);
        STOP_TIMER(TIMER_FINE);

        // allocate buffer
        TIMED_SCOPE("Allocation", TIMER_FINE) {
            _gain_buffer.resize(_gain_buffer_index[_graph->n() - 1]);
        };

        START_TIMER("Build buffer", TIMER_FINE);
        shm::parallel::chunked_for(join_requests_per_pe, [&](const JoinRequest& request) {
            KASSERT(_graph->is_owned_global_node(request.global_requested));
            const NodeID local_requested = _graph->global_to_local_node(request.global_requested);
            KASSERT(local_requested < _gain_buffer_index.size());
            const NodeID global_requester = request.global_requester;

            KASSERT(_gain_buffer_index[local_requested] - 1 < _gain_buffer.size());
            _gain_buffer[--_gain_buffer_index[local_requested]] = {
                global_requester, request.requester_weight, request.requester_gain};
        });
        STOP_TIMER(TIMER_FINE);

        if constexpr (kDebug) {
            std::ostringstream gain_buffer_str;
            gain_buffer_str << "UNSORTED\n";
            for (const auto& entry: _gain_buffer) {
                gain_buffer_str << "{.global_node=" << entry.global_node << ", .node_weight=" << entry.node_weight
                                << ", .gain=" << entry.gain << "} ";
            }
        }

        // sort buffer for each node by gain
        START_TIMER("Sort buffer", TIMER_FINE);
        tbb::parallel_for<NodeID>(0, _graph->n(), [&](const NodeID u) {
            KASSERT(u + 1 < _gain_buffer_index.size());
            if (_gain_buffer_index[u] < _gain_buffer_index[u + 1]) {
                std::sort(
                    _gain_buffer.begin() + _gain_buffer_index[u], _gain_buffer.begin() + _gain_buffer_index[u + 1],
                    [&](const auto& lhs, const auto& rhs) {
                        KASSERT(lhs.global_node != rhs.global_node);
                        const double lhs_rel_gain = 1.0 * lhs.gain / lhs.node_weight;
                        const double rhs_rel_gain = 1.0 * rhs.gain / rhs.node_weight;
                        return lhs_rel_gain < rhs_rel_gain
                               || (lhs_rel_gain == rhs_rel_gain && lhs.global_node < rhs.global_node);
                    });
            }
        });
        STOP_TIMER(TIMER_FINE);

        if constexpr (kDebug) {
            std::ostringstream gain_buffer_str;
            gain_buffer_str << "SORTED\n";
            for (const auto& entry: _gain_buffer) {
                gain_buffer_str << "{.global_node=" << entry.global_node << ", .node_weight=" << entry.node_weight
                                << ", .gain=" << entry.gain << "} ";
            }
            SLOG << gain_buffer_str.str();
        }
    }

    //! Synchronize labels of ghost nodes.
    void synchronize_labels(const NodeID from, const NodeID to) {
        SCOPED_TIMER("Synchronize labels");

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
        KASSERT((u < _next_clustering.size() && u < _current_clustering.size()));
        return _next_clustering[u] != _current_clustering[u];
    }

#if KASSERT_ENABLED(ASSERTION_LEVEL_HEAVY)
    bool VALIDATE_INIT_STATE() {
        KASSERT(_graph->total_n() <= _current_clustering.size());
        KASSERT(_graph->total_n() <= _next_clustering.size());
        KASSERT(_graph->n() <= _locked.size());
        KASSERT(_graph->n() <= _gain.size());

        for (const NodeID u: _graph->all_nodes()) {
            KASSERT(_current_clustering[u] == _next_clustering[u]);
            KASSERT(cluster(u) == _graph->local_to_global_node(u));
            KASSERT(cluster_weight(cluster(u)) == _graph->node_weight(u));
        }

        return true;
    }

    bool VALIDATE_STATE() {
        for (const NodeID u: _graph->all_nodes()) {
            KASSERT(_current_clustering[u] == _next_clustering[u]);
        }
        KASSERT(VALIDATE_LOCKING_INVARIANT());
        return true;
    }

    bool VALIDATE_LOCKING_INVARIANT() {
        mpi::graph::sparse_alltoall_custom<GlobalNodeID>(
            *_graph, 0, _graph->n(),
            [&](const NodeID u) {
                KASSERT(cluster(u) < _graph->global_n());
                return !_graph->is_owned_global_node(cluster(u));
            },
            [&](const NodeID u) { return _graph->find_owner_of_global_node(cluster(u)); },
            [&](const NodeID u) { return cluster(u); },
            [&](const auto& buffer, const PEID pe) {
                for (const GlobalNodeID label: buffer) {
                    KASSERT(_graph->is_owned_global_node(label));
                    const NodeID local_label = _graph->global_to_local_node(label);
                    KASSERT(
                        cluster(local_label) == label, "from PE: " << pe << " has nodes in cluster " << label
                                                                   << ", but local node " << local_label
                                                                   << " is in cluster " << cluster(local_label));
                    KASSERT(_locked[local_label] == 1);
                }
            });

        return true;
    }
#endif

    using Base::_graph;

    const CoarseningContext& _c_ctx;

    NodeWeight                  _max_cluster_weight{kInvalidNodeWeight};
    AtomicClusterArray          _current_clustering;
    AtomicClusterArray          _next_clustering;
    scalable_vector<EdgeWeight> _gain;

    struct GainBufferEntry {
        GlobalNodeID global_node;
        NodeWeight   node_weight;
        EdgeWeight   gain;
    };

    //! After receiving join requests, sort ghost nodes that want to join a cluster here. Use \c _gain_buffer_index to
    //! navigate this vector.
    scalable_vector<GainBufferEntry> _gain_buffer;
    //! After receiving join requests, sort ghost nodes that want to join a cluster into \c _gain_buffer. For each
    //! interface node, store the index for that nodes join requests in \c _gain_buffer in this vector.
    scalable_vector<Atomic<NodeID>> _gain_buffer_index;

    scalable_vector<std::uint8_t> _locked;

    using ClusterWeightsMap = typename growt::GlobalNodeIDMap<GlobalNodeWeight>;
    ClusterWeightsMap                                                        _cluster_weights{0};
    tbb::enumerable_thread_specific<typename ClusterWeightsMap::handle_type> _cluster_weights_handles_ets{[&] {
        return ClusterWeightsMap::handle_type{_cluster_weights};
    }};
};

LockingLabelPropagationClustering::LockingLabelPropagationClustering(const Context& ctx)
    : _impl{std::make_unique<LockingLabelPropagationClusteringImpl>(ctx)} {}

LockingLabelPropagationClustering::~LockingLabelPropagationClustering() = default;

const LockingLabelPropagationClustering::AtomicClusterArray& LockingLabelPropagationClustering::compute_clustering(
    const DistributedGraph& graph, const GlobalNodeWeight max_cluster_weight) {
    return _impl->compute_clustering(graph, max_cluster_weight);
}
} // namespace dkaminpar
