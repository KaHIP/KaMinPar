/*******************************************************************************
 * @file:   global_label_propagation_coarsener.cc
 * @author: Daniel Seemaier
 * @date:   29.09.21
 * @brief:  Label propagation with clusters that span multiple PEs. Cluster
 * labels and weights are synchronized in rounds. Between communication rounds,
 * a cluster can grow beyond the maximum cluster weight limit if more than one
 * PE moves nodes to the cluster. Thus, the clustering might violate the
 * maximum cluster weight limit.
 ******************************************************************************/
#include "dkaminpar/coarsening/global_label_propagation_clustering.h"

#include <unordered_map>

#include "datastructure/distributed_graph.h"
#include "datastructure/fast_reset_array.h"
#include "dkaminpar/growt.h"
#include "dkaminpar/mpi_graph.h"
#include "dkaminpar/utils/math.h"
#include "kaminpar/label_propagation.h"

namespace dkaminpar {
namespace {
inline const DistributedGraph* _global_graph;

template <typename Iter1, typename Iter2>
class CombinedRange {
public:
    class iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type        = typename Iter1::value_type;
        using difference_type   = std::make_signed_t<std::size_t>;
        using pointer           = value_type*;
        using reference         = value_type&;

        iterator(Iter1 begin1, std::size_t size1, Iter2 begin2, std::size_t size2, bool end)
            : _begin1(begin1),
              _size1(size1),
              _begin2(begin2),
              _size2(size2) {
            if (end) {
                _second = true;
                _index  = _size2;
            }
        }

        value_type operator*() const {
            KASSERT((!_second && _index < _size1) || (_second && _index < _size2));
            if (_second) {
                return *_begin2;
            } else {
                const auto& [node, gain] = *_begin1;
                return std::make_pair(_global_graph->local_to_global_node(node), gain);
            }
        }

        iterator& operator++() {
            ++_index;
            if (!_second && _index < _size1) {
                ++_begin1;
            } else if (!_second && _index == _size1) {
                _second = true;
                _index  = 0;
            } else {
                ++_begin2;
            }
            return *this;
        }

        iterator operator++(int) {
            auto tmp = *this;
            ++*this;
            return tmp;
        }

        bool operator==(const iterator& other) const {
            return _index == other._index && _second == other._second;
        }
        bool operator!=(const iterator& other) const {
            return _index != other._index || _second != other._second;
        }

    private:
        Iter1       _begin1;
        std::size_t _size1;
        Iter2       _begin2;
        std::size_t _size2;
        bool        _second = false;
        std::size_t _index  = 0;
    };

    CombinedRange(Iter1 begin1, std::size_t size1, Iter2 begin2, std::size_t size2)
        : _begin(begin1, size1, begin2, size2, false),
          _end(begin1, size1, begin2, size2, true) {}

    iterator begin() const {
        return _begin;
    }
    iterator end() const {
        return _end;
    }

private:
    iterator _begin;
    iterator _end;
};

struct VectorHashRatingMap {
    VectorHashRatingMap() {}

    EdgeWeight& operator[](const GlobalNodeID key) {
        KASSERT(_global_graph != nullptr);
        if (_global_graph->is_owned_global_node(key)) {
            KASSERT(_global_graph->global_to_local_node(key) < _local.capacity());
            return _local[_global_graph->global_to_local_node(key)];
        } else {
            return _global[key];
        }
    }

    [[nodiscard]] auto entries() {
        return CombinedRange(_local.entries().begin(), _local.size(), _global.begin(), _global.size());
    }

    void clear() {
        _local.clear();
        _global.clear();
    }

    std::size_t capacity() const {
        return _local.capacity();
    }

    void resize(const std::size_t capacity) {
        _local.resize(capacity);
    }

    shm::FastResetArray<EdgeWeight>              _local{};
    std::unordered_map<GlobalNodeID, EdgeWeight> _global{};
};

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

    void resize(const std::size_t /* capacity */) {}

    std::unordered_map<GlobalNodeID, EdgeWeight> map{};
};

struct DistributedGlobalLabelPropagationClusteringConfig : public shm::LabelPropagationConfig {
    using Graph = DistributedGraph;
    // using RatingMap = ::kaminpar::RatingMap<EdgeWeight, VectorHashRatingMap>;
    using RatingMap                            = ::kaminpar::RatingMap<EdgeWeight, UnorderedRatingMap>;
    using ClusterID                            = GlobalNodeID;
    using ClusterWeight                        = GlobalNodeWeight;
    static constexpr bool kTrackClusterCount   = false;
    static constexpr bool kUseTwoHopClustering = false;
};
} // namespace

class DistributedGlobalLabelPropagationClusteringImpl final
    : public shm::ChunkRandomizedLabelPropagation<
          DistributedGlobalLabelPropagationClusteringImpl, DistributedGlobalLabelPropagationClusteringConfig>,
      public shm::OwnedClusterVector<NodeID, GlobalNodeID> {
    SET_DEBUG(false);

    using Base = shm::ChunkRandomizedLabelPropagation<
        DistributedGlobalLabelPropagationClusteringImpl, DistributedGlobalLabelPropagationClusteringConfig>;
    using ClusterBase = shm::OwnedClusterVector<NodeID, GlobalNodeID>;

public:
    explicit DistributedGlobalLabelPropagationClusteringImpl(const Context& ctx)
        : ClusterBase{ctx.partition.total_n()},
          _c_ctx{ctx.coarsening},
          _changed_label(ctx.partition.local_n()),
          _cluster_weights{ctx.partition.total_n() - ctx.partition.local_n()},
          _local_cluster_weights(ctx.partition.local_n()) {
        set_max_num_iterations(_c_ctx.global_lp.num_iterations);
        set_max_degree(_c_ctx.global_lp.large_degree_threshold);
        set_max_num_neighbors(_c_ctx.global_lp.max_num_neighbors);
    }

    const auto& compute_clustering(const DistributedGraph& graph, const GlobalNodeWeight max_cluster_weight) {
        SCOPED_TIMER("Label propagation");

        _global_graph = &graph;
        {
            SCOPED_TIMER("Allocation", TIMER_DETAIL);
            allocate(graph);
        }

        {
            SCOPED_TIMER("Initialization", TIMER_DETAIL);

            // clear hash map
            _cluster_weights_handles_ets.clear();
            _cluster_weights = ClusterWeightsMap{0};
            std::fill(_local_cluster_weights.begin(), _local_cluster_weights.end(), 0);

            // initialize data structures
            initialize(&graph, graph.total_n());
            initialize_ghost_node_clusters();
            _max_cluster_weight = max_cluster_weight;
        }

        for (std::size_t iteration = 0; iteration < _max_num_iterations; ++iteration) {
            GlobalNodeID global_num_moved_nodes = 0;
            for (std::size_t chunk = 0; chunk < _c_ctx.global_lp.num_chunks; ++chunk) {
                const auto [from, to] =
                    math::compute_local_range<NodeID>(_graph->n(), _c_ctx.global_lp.num_chunks, chunk);
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
     * Note: offset cluster IDs by 1 since growt cannot use 0 as key.
     */

    void init_cluster_weight(const ClusterID local_cluster, const ClusterWeight weight) {
        if (_graph->is_owned_node(local_cluster)) {
            _local_cluster_weights[local_cluster] = weight;
        } else {
            KASSERT(local_cluster < _graph->total_n());
            const auto cluster = _graph->local_to_global_node(static_cast<NodeID>(local_cluster));

            auto& handle                              = _cluster_weights_handles_ets.local();
            [[maybe_unused]] const auto [it, success] = handle.insert(cluster + 1, weight);
            KASSERT(success, "Cluster already initialized: " << cluster + 1);
        }
    }

    ClusterWeight cluster_weight(const ClusterID cluster) {
        if (_graph->is_owned_global_node(cluster)) {
            return _local_cluster_weights[_graph->global_to_local_node(cluster)];
        } else {
            auto& handle = _cluster_weights_handles_ets.local();
            auto  it     = handle.find(cluster + 1);
            KASSERT(it != handle.end(), "Uninitialized cluster: " << cluster + 1);
            return (*it).second;
        }
    }

    bool move_cluster_weight(
        const ClusterID old_cluster, const ClusterID new_cluster, const ClusterWeight delta,
        const ClusterWeight max_weight) {
        // reject move if it violates local weight constraint
        if (cluster_weight(new_cluster) + delta > max_weight) {
            return false;
        }

        auto& handle = _cluster_weights_handles_ets.local();

        if (_graph->is_owned_global_node(old_cluster)) {
            _local_cluster_weights[_graph->global_to_local_node(old_cluster)] -= delta;
        } else {
            // otherwise, move node to new cluster
            [[maybe_unused]] const auto [old_it, old_found] = handle.update(
                old_cluster + 1, [](auto& lhs, const auto rhs) { return lhs -= rhs; }, delta);
            KASSERT((old_it != handle.end() && old_found), "Uninitialized cluster: " << old_cluster + 1);
        }

        if (_graph->is_owned_global_node(new_cluster)) {
            _local_cluster_weights[_graph->global_to_local_node(new_cluster)] += delta;
        } else {
            [[maybe_unused]] const auto [new_it, new_found] = handle.update(
                new_cluster + 1, [](auto& lhs, const auto rhs) { return lhs += rhs; }, delta);
            KASSERT((new_it != handle.end() && new_found), "Uninitialized cluster: " << new_cluster + 1);
        }

        return true;
    }

    void
    change_cluster_weight(const ClusterID cluster, const ClusterWeight delta, [[maybe_unused]] const bool must_exist) {
        if (_graph->is_owned_global_node(cluster)) {
            _local_cluster_weights[_graph->global_to_local_node(cluster)] += delta;
        } else {
            auto& handle                                = _cluster_weights_handles_ets.local();
            [[maybe_unused]] const auto [it, not_found] = handle.insert_or_update(
                cluster + 1, delta, [](auto& lhs, const auto rhs) { return lhs += rhs; }, delta);
            KASSERT((it != handle.end() && (!must_exist || !not_found)), "Could not update cluster: " << cluster);
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

    void move_node(const NodeID node, const ClusterID cluster) {
        KASSERT(node < _changed_label.size());
        OwnedClusterVector::move_node(node, cluster);
        _changed_label[node] = 1;
    }

    [[nodiscard]] ClusterID initial_cluster(const NodeID u) {
        return _graph->local_to_global_node(u);
    }

    /*
     * Moving nodes
     */

    [[nodiscard]] bool accept_cluster(const Base::ClusterSelectionState& state) {
        return (state.current_gain > state.best_gain
                || (state.current_gain == state.best_gain && state.local_rand.random_bool()))
               && (state.current_cluster_weight + state.u_weight <= max_cluster_weight(state.current_cluster)
                   || state.current_cluster == state.initial_cluster);
    }

    [[nodiscard]] inline bool activate_neighbor(const NodeID u) {
        return _graph->is_owned_node(u);
    }
    //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //
    // Called from base class
    //
    //--------------------------------------------------------------------------------

private:
    void allocate(const DistributedGraph& graph) {
        ensure_cluster_size(graph.total_n());

        const NodeID allocated_num_active_nodes = _changed_label.size();

        if (allocated_num_active_nodes < graph.n()) {
            _changed_label.resize(graph.n());
            _local_cluster_weights.resize(graph.n());
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
        START_TIMER("Chunk iteration", TIMER_DETAIL);
        const NodeID local_num_moved_nodes = perform_iteration(from, to);
        STOP_TIMER(TIMER_DETAIL);

        const GlobalNodeID global_num_moved_nodes =
            mpi::allreduce(local_num_moved_nodes, MPI_SUM, _graph->communicator());

        if (global_num_moved_nodes > 0) {
            synchronize_ghost_node_clusters(from, to);
        }

        if (_c_ctx.global_lp.merge_singleton_clusters) {
            cluster_isolated_nodes(from, to);
        }

        return global_num_moved_nodes;
    }

    void synchronize_ghost_node_clusters(const NodeID from, const NodeID to) {
        SCOPED_TIMER("Synchronize ghost node clusters", TIMER_DETAIL);

        struct ChangedLabelMessage {
            NodeID    local_node;
            ClusterID new_label;
        };

        mpi::graph::sparse_alltoall_interface_to_pe<ChangedLabelMessage>(
            *_graph, from, to, [&](const NodeID u) { return _changed_label[u]; },
            [&](const NodeID u) -> ChangedLabelMessage {
                return {u, cluster(u)};
            },
            [&](const auto& buffer, const PEID pe) {
                tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
                    const auto [local_node_on_pe, new_label] = buffer[i];

                    const GlobalNodeID global_node = _graph->offset_n(pe) + local_node_on_pe;
                    KASSERT(!_graph->is_owned_global_node(global_node));

                    const NodeID     local_node        = _graph->global_to_local_node(global_node);
                    const NodeWeight local_node_weight = _graph->node_weight(local_node);

                    change_cluster_weight(cluster(local_node), -local_node_weight, true);
                    OwnedClusterVector::move_node(local_node, new_label);
                    change_cluster_weight(cluster(local_node), local_node_weight, false);
                });
            });

        _graph->pfor_nodes(from, to, [&](const NodeID u) { _changed_label[u] = 0; });
    }

    /*!
     * Build clusters of isolated nodes: store the first isolated node and add subsequent isolated nodes to its cluster
     * until the maximum cluster weight is violated; then, move on to the next isolated node etc.
     * @param from The first node to consider.
     * @param to One-after the last node to consider.
     */
    void cluster_isolated_nodes(const NodeID from, const NodeID to) {
        SCOPED_TIMER("Cluster isolated nodes", TIMER_DETAIL);

        tbb::enumerable_thread_specific<GlobalNodeID> isolated_node_ets(kInvalidNodeID);
        tbb::parallel_for(tbb::blocked_range<NodeID>(from, to), [&](tbb::blocked_range<NodeID> r) {
            NodeID        current         = isolated_node_ets.local();
            ClusterID     current_cluster = current == kInvalidNodeID ? kInvalidGlobalNodeID : cluster(current);
            ClusterWeight current_weight =
                current == kInvalidNodeID ? kInvalidNodeWeight : cluster_weight(current_cluster);

            for (NodeID u = r.begin(); u != r.end(); ++u) {
                if (_graph->degree(u) == 0) {
                    const auto u_cluster = cluster(u);
                    const auto u_weight  = cluster_weight(u_cluster);

                    if (current != kInvalidNodeID && current_weight + u_weight <= max_cluster_weight(u_cluster)) {
                        change_cluster_weight(current_cluster, u_weight, true);
                        OwnedClusterVector::move_node(u, current_cluster);
                        current_weight += u_weight;
                    } else {
                        current         = u;
                        current_cluster = u_cluster;
                        current_weight  = u_weight;
                    }
                }
            }

            isolated_node_ets.local() = current;
        });
    }

    using Base::_graph;
    const CoarseningContext& _c_ctx;
    NodeWeight               _max_cluster_weight{std::numeric_limits<NodeWeight>::max()};
    std::size_t              _max_num_iterations{std::numeric_limits<std::size_t>::max()};

    //! \code{_changed_label[u] = 1} iff. node \c u changed its label in the current round
    scalable_vector<uint8_t> _changed_label;

    using ClusterWeightsMap = typename growt::GlobalNodeIDMap<GlobalNodeWeight>;
    ClusterWeightsMap                                                        _cluster_weights{0};
    tbb::enumerable_thread_specific<typename ClusterWeightsMap::handle_type> _cluster_weights_handles_ets{[&] {
        return ClusterWeightsMap::handle_type{_cluster_weights};
    }};
    scalable_vector<GlobalNodeWeight>                                        _local_cluster_weights;
};

//
// Exposed wrapper
//

DistributedGlobalLabelPropagationClustering::DistributedGlobalLabelPropagationClustering(const Context& ctx)
    : _impl{std::make_unique<DistributedGlobalLabelPropagationClusteringImpl>(ctx)} {}

DistributedGlobalLabelPropagationClustering::~DistributedGlobalLabelPropagationClustering() = default;

const DistributedGlobalLabelPropagationClustering::AtomicClusterArray&
DistributedGlobalLabelPropagationClustering::compute_clustering(
    const DistributedGraph& graph, const GlobalNodeWeight max_cluster_weight) {
    return _impl->compute_clustering(graph, max_cluster_weight);
}
} // namespace dkaminpar
