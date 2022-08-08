/*******************************************************************************
 * @file:   fm_refiner.cc
 * @author: Daniel Seemaier
 * @date:   02.08.2022
 * @brief:  Distributed FM refiner.
 ******************************************************************************/
#include "dkaminpar/refinement/fm_refiner.h"

#include <cstdint>
#include <limits>
#include <random>
#include <stack>

#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>

#include "dkaminpar/context.h"
#include "dkaminpar/graphutils/graph_synchronization.h"
#include "dkaminpar/growt.h"
#include "dkaminpar/mpi/alltoall.h"
#include "dkaminpar/mpi/utils.h"

#include "kaminpar/datastructure/graph.h"

#include "common/datastructures/binary_heap.h"
#include "common/datastructures/marker.h"
#include "common/datastructures/rating_map.h"
#include "common/noinit_vector.h"
#include "common/parallel/atomic.h"
#include "common/random.h"
#include "common/timer.h"

namespace kaminpar::dist {
SET_DEBUG(true);

namespace {
struct AdaptiveStoppingPolicy {
    AdaptiveStoppingPolicy(const GlobalNodeID n) : _beta(std::log(n)) {}

    [[nodiscard]] bool should_stop(const double alpha) const {
        const double factor = alpha / 2.0 - 0.25;
        return (_num_steps > _beta) && ((_mk == 0.0) || (_num_steps >= (_variance / (_mk * _mk)) * factor));
    }

    void reset() {
        _num_steps = 0;
        _variance  = 0.0;
    }

    void update(const EdgeWeight gain) {
        ++_num_steps;

        // see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        if (_num_steps == 1) {
            _mk_minus_one = 1.0 * gain;
            _mk           = _mk_minus_one;
            _sk_minus_one = 0.0;
        } else {
            _mk           = _mk_minus_one + (gain - _mk_minus_one) / _num_steps;
            _sk           = _sk_minus_one + (gain - _mk_minus_one) * (gain - _mk);
            _variance     = _sk / (_num_steps - 1.0);
            _mk_minus_one = _mk;
            _sk_minus_one = _sk;
        }
    }

private:
    double      _beta{0.0};
    std::size_t _num_steps{0};
    double      _variance{0.0};
    double      _mk{0.0};
    double      _mk_minus_one{0.0};
    double      _sk{0.0};
    double      _sk_minus_one{0.0};
};

class LocalFMRefiner {
public:
    struct Move {
        Move(const NodeID node, const BlockID block) : node(node), block(block) {}

        NodeID  node;
        BlockID block;
    };

    LocalFMRefiner(
        const DistributedPartitionedGraph& global_graph, const FMRefinementContext& fm_ctx,
        const PartitionContext& p_ctx)
        : _global_graph(global_graph),
          _fm_ctx(fm_ctx),
          _p_ctx(p_ctx),
          _rating_map(_p_ctx.k),
          _stopping_policy(p_ctx.global_n()) {}

    std::vector<Move> refine(shm::PartitionedGraph& p_graph, const std::vector<bool>& fixed_nodes) {
        if (p_graph.n() == 0) {
            return {};
        }

        initialize(p_graph, fixed_nodes);

        // record of FM nodes for rollback
        EdgeWeight        best_total_gain    = 0;
        EdgeWeight        current_total_gain = 0;
        std::size_t       rollback_index     = 0;
        std::vector<Move> moves;

        while (!_pq.empty() && !_stopping_policy.should_stop(_fm_ctx.alpha)) {
            // retrieve next node from PQ
            const NodeID     u      = _pq.peek_id();
            const NodeWeight weight = _p_graph->node_weight(u);
            const BlockID    from   = _p_graph->block(u);
            const auto [gain, to]   = find_best_target_block<false>(u);
            _pq.pop();

            // only perform move if target block can take u without becoming overloaded
            const bool feasible =
                to != from
                && _global_graph.block_weight(to) + _block_weight_deltas[to] + weight <= _p_ctx.max_block_weight(to);

            if (feasible) {
                // move u to its target block
                _p_graph->set_block<false>(u, to);
                _block_weight_deltas[from] -= weight;
                _block_weight_deltas[to] += weight;
                moves.emplace_back(u, from);
                update_pq_after_move(u, from, to, fixed_nodes);
                current_total_gain += gain;
                _stopping_policy.update(gain);

                if (current_total_gain > best_total_gain) {
                    best_total_gain = current_total_gain;
                    rollback_index  = moves.size();
                    _stopping_policy.reset();
                }
            } else {
                // target block became too full since PQ initialization
                // --> retry with new target block
                insert_node_into_pq<false>(u);
            }
        }

        // rollback to best cut found
        while (moves.size() > rollback_index) {
            const auto [node, to]   = moves.back();
            const BlockID    from   = _p_graph->block(node);
            const NodeWeight weight = _p_graph->node_weight(node);

            _block_weight_deltas[from] -= weight;
            _block_weight_deltas[to] += weight;
            _p_graph->set_block<false>(node, to);

            moves.pop_back();
        }

        // for the moves that we keep, store the new block rather than the old one
        for (auto& [u, block]: moves) {
            block = _p_graph->block(u);
        }

        //DBG << "Improved local cut by " << best_total_gain;
        return moves;
    }

    void commit_block_weight_deltas(DistributedPartitionedGraph& p_graph) {
        for (const auto& [block, delta]: _block_weight_deltas) {
            p_graph.set_block_weight(block, p_graph.block_weight(block) + delta);
        }
    }

private:
    void initialize(shm::PartitionedGraph& p_graph, const std::vector<bool> fixed_nodes) {
        _p_graph = &p_graph;

        // resize data structures s.t. they are large enough for _p_graph
        if (_pq.capacity() < _p_graph->n()) {
            _pq.clear();
            _pq.resize(_p_graph->n());
            _marker.resize(_p_graph->n());
        }

        // clear data structures from previous run
        _marker.reset();
        _pq.clear();
        _stopping_policy.reset();
        _block_weight_deltas.clear();

        // fill PQ with all border nodes
        for (const NodeID u: _p_graph->nodes()) {
            if (!fixed_nodes[u]) {
                insert_node_into_pq<true>(u);
            }
        }
    }

    void
    update_pq_after_move(const NodeID u, const BlockID from, const BlockID to, const std::vector<bool>& fixed_nodes) {
        // update neighbors
        for (const auto [e, v]: _p_graph->neighbors(u)) {
            if (fixed_nodes[v]) {
                continue;
            }

            const BlockID v_block = _p_graph->block(v);

            if (v_block == from || v_block == to) {
                const auto [best_gain, best_target_block] = find_best_target_block<false>(v);
                if (_pq.contains(v)) {
                    _pq.change_priority(v, best_gain);
                } else if (!_marker.get(v)) {
                    insert_node_into_pq<false>(v);
                    _marker.set(v);
                }
            }
        }
    }

    template <bool initialization>
    void insert_node_into_pq(const NodeID u) {
        const auto [best_gain, best_target_block] = find_best_target_block<initialization>(u);
        if (_p_graph->block(u) != best_target_block) {
            _pq.push(u, best_gain);
            if (initialization) {
                _marker.set(u);
            }
        }
    }

    template <bool initialization>
    std::pair<EdgeWeight, BlockID> find_best_target_block(const NodeID u) {
        const BlockID u_block = _p_graph->block(u);

        auto action = [&](auto& map) {
            EdgeWeight internal_degree = 0;
            for (const auto [e, v]: _p_graph->neighbors(u)) {
                const BlockID    v_block  = _p_graph->block(v);
                const EdgeWeight e_weight = _p_graph->edge_weight(e);

                if (v_block != u_block) {
                    map[v_block] += e_weight;
                } else {
                    internal_degree += e_weight;
                }
            }

            EdgeWeight best_gain         = std::numeric_limits<EdgeWeight>::min();
            BlockID    best_target_block = u_block;

            const NodeWeight u_weight = _p_graph->node_weight(u);

            for (const auto [current_target_block, current_gain]: map.entries()) {
                // compute weight of block if we were to move the node
                BlockWeight block_weight_prime = _global_graph.block_weight(current_target_block) + u_weight;
                if (!initialization) {
                    block_weight_prime += _block_weight_deltas[current_target_block];
                }
                const bool feasible = block_weight_prime <= _p_ctx.max_block_weight(current_target_block);

                // accept as better block if gain is larger
                // if gain is equal, flip a coin
                if (feasible && (current_gain > best_gain || (current_gain == best_gain && _rand.random_bool()))) {
                    best_gain         = current_gain;
                    best_target_block = current_target_block;
                }
            }

            // subtract internal degree to get the actual gain value of this move
            best_gain -= internal_degree; // overflow OK, value unused if still set to min

            map.clear(); // clear for next node
            return std::make_pair(best_gain, best_target_block);
        };

        _rating_map.update_upper_bound_size(std::min<BlockID>(_p_ctx.k, _p_graph->degree(u)));
        return _rating_map.run_with_map(action, action);
    }

    // initialized by ctor
    const DistributedPartitionedGraph& _global_graph;
    const FMRefinementContext&         _fm_ctx;
    const PartitionContext&            _p_ctx;
    RatingMap<EdgeWeight, NodeID>      _rating_map;
    AdaptiveStoppingPolicy             _stopping_policy;

    // initialized by initialize_refiner()
    shm::PartitionedGraph*    _p_graph;
    BinaryMaxHeap<EdgeWeight> _pq{0};
    Marker<>                  _marker{0};

    // initialized right here
    Random&                                  _rand{Random::instance()};
    std::unordered_map<BlockID, BlockWeight> _block_weight_deltas;
};
} // namespace

FMRefiner::FMRefiner(const Context& ctx) : _p_ctx(ctx.partition), _fm_ctx(ctx.refinement.fm) {}

void FMRefiner::initialize(const DistributedGraph&, const PartitionContext&) {}

void FMRefiner::refine(DistributedPartitionedGraph& p_graph) {
    SCOPED_TIMER("FM");
    _p_graph = &p_graph;
    ++_iteration;

    // collect seed nodes
    const auto seed_nodes = find_seed_nodes();

    std::vector<shm::Graph>                     local_graphs(seed_nodes.size());
    std::vector<shm::PartitionedGraph>          p_local_graphs(seed_nodes.size());
    std::vector<std::vector<GlobalNodeID>>      local_graph_mappings(seed_nodes.size());
    std::vector<parallel::Atomic<std::uint8_t>> taken(p_graph.total_n());
    std::vector<std::vector<bool>>              fixed_nodes(seed_nodes.size());
    using GlobalMovesMap = growt::GlobalNodeIDMap<BlockID>;
    GlobalMovesMap                                                        global_moves(0);
    tbb::enumerable_thread_specific<typename GlobalMovesMap::handle_type> global_moves_handle_ets{[&] {
        return GlobalMovesMap::handle_type{global_moves};
    }};

    START_TIMER("Collect local search graphs");
    // mark seed nodes as taken
    tbb::parallel_for<std::size_t>(0, seed_nodes.size(), [&](const std::size_t i) { taken[seed_nodes[i]] = 1; });

    struct DiscoveredNode {
        DiscoveredNode() {}
        DiscoveredNode(const NodeID node, const NodeID distance, const bool border)
            : node(node),
              distance(distance),
              border(border) {}

        NodeID node;
        NodeID distance;
        bool   border;
    };

    tbb::parallel_for<std::size_t>(0, seed_nodes.size(), [&](const std::size_t i) {
        const NodeID                seed_node = seed_nodes[i];
        std::vector<DiscoveredNode> discovered_owned_nodes;
        std::vector<DiscoveredNode> discovered_ghost_nodes;

        std::stack<NodeID> search_front;
        search_front.push(seed_node);
        NodeID current_front_size = 1; // no. of elements beloning to current front
        NodeID current_distance   = 0;

        while (current_distance < _fm_ctx.distance + 1 && !search_front.empty()) {
            const NodeID current = search_front.top();
            search_front.pop();
            --current_front_size;

            // try to take this node
            if (_p_graph->is_owned_node(current)) {
                discovered_owned_nodes.emplace_back(current, current_distance, false);

                // grow to neighbors
                for (const auto [e, v]: _p_graph->neighbors(current)) {
                    std::uint8_t free = 0;
                    if (current_distance + 1 < _fm_ctx.distance && taken[v].compare_exchange_strong(free, 1)) {
                        search_front.push(v);
                    } else {
                        discovered_owned_nodes.emplace_back(v, current_distance + 1, true);
                    }
                }
            } else {
                discovered_owned_nodes.emplace_back(current, current_distance, true);
                discovered_ghost_nodes.emplace_back(current, current_distance, false);
            }

            // +1 distance from seed
            if (current_front_size == 0) {
                current_front_size = search_front.size();
                ++current_distance;
            }
        }

        // @todo inter-PE growth

        // build local graphs
        const NodeID n = discovered_owned_nodes.size();

        // build local_graph_mappings[i]
        auto& mapping = local_graph_mappings[i];
        auto& fixed   = fixed_nodes[i];
        for (const auto [u, d, b]: discovered_owned_nodes) {
            mapping.push_back(_p_graph->local_to_global_node(u));
            fixed.push_back(b);
        }

        // build inverse mapping
        std::unordered_map<GlobalNodeID, NodeID> to_local_map;
        for (std::size_t i = 0; i < n; ++i) {
            to_local_map[mapping[i]] = i;
        }

        // build local graph
        StaticArray<EdgeID>     local_nodes(n + 1);
        std::vector<NodeID>     local_edges;
        StaticArray<NodeWeight> local_node_weights(n);
        std::vector<EdgeWeight> local_edge_weights;
        NodeID                  next_node = 0;

        for (const auto [u, d, b]: discovered_owned_nodes) {
            local_nodes[next_node]        = local_edges.size();
            local_node_weights[next_node] = _p_graph->node_weight(u);
            ++next_node;

            for (const auto [e, v]: _p_graph->neighbors(u)) {
                const GlobalNodeID global_v = _p_graph->local_to_global_node(v);
                if (to_local_map.find(global_v) != to_local_map.end()) {
                    local_edges.push_back(to_local_map[global_v]);
                    local_edge_weights.push_back(_p_graph->edge_weight(e));
                }
            }
        }
        local_nodes[n] = local_edges.size();

        // build partition
        StaticArray<BlockID> partition(n);
        for (std::size_t i = 0; i < discovered_owned_nodes.size(); ++i) {
            partition[i] = _p_graph->block(discovered_owned_nodes[i].node);
        }

        // create graph objects
        StaticArray<NodeID> local_edges_prime(local_edges.size());
        std::copy(local_edges.begin(), local_edges.end(), local_edges_prime.begin());
        StaticArray<EdgeWeight> local_edge_weights_prime(local_edge_weights.size());
        std::copy(local_edge_weights.begin(), local_edge_weights.end(), local_edge_weights_prime.begin());

        local_graphs[i] = shm::Graph(
            std::move(local_nodes), std::move(local_edges_prime), std::move(local_node_weights),
            std::move(local_edge_weights_prime));
        p_local_graphs[i] =
            shm::PartitionedGraph(shm::no_block_weights, local_graphs[i], _p_ctx.k, std::move(partition));

        //DBG << "Local graph around seed node " << i << " grew to " << local_graphs[i].n() << " nodes and "
        //    << local_graphs[i].m() << " edges";
    });
    STOP_TIMER();

    START_TIMER("Local search");
    tbb::enumerable_thread_specific<LocalFMRefiner> local_refiner_ets(
        [&] { return LocalFMRefiner(*_p_graph, _fm_ctx, _p_ctx); });

    tbb::parallel_for<std::size_t>(0, local_graphs.size(), [&](const std::size_t i) {
        auto&       p_local_graph     = p_local_graphs[i];
        const auto& local_fixed_nodes = fixed_nodes[i];
        auto&       refiner           = local_refiner_ets.local();
        const auto  moves             = refiner.refine(p_local_graph, local_fixed_nodes);
        refiner.commit_block_weight_deltas(*_p_graph);
        auto&       global_moves_handle = global_moves_handle_ets.local();
        const auto& mapping             = local_graph_mappings[i];

        for (const auto [u, to]: moves) {
            const GlobalNodeID global_u               = mapping[u];
            const GlobalNodeID global_u_prime         = global_u + 1; // growt does not allow 0 as key
            [[maybe_unused]] const auto [it, success] = global_moves_handle.insert(global_u_prime, to);
            KASSERT(success);
        }
    });
    STOP_TIMER();

    START_TIMER("Broadcast moves");
    struct MoveMessage {
        MoveMessage() {}
        MoveMessage(const NodeID node, const BlockID block) : node(node), block(block) {}
        NodeID  node  = 0;
        BlockID block = 0;
    };

    START_TIMER("Build send buffer");
    std::vector<std::vector<MoveMessage>> sendbuf(mpi::get_comm_size(_p_graph->communicator()));
    for (const auto [to, global_u_prime]: global_moves_handle_ets.local()) { // @todo parallelize
        KASSERT(global_u_prime != 0u);
        const GlobalNodeID global_u = global_u_prime - 1;
        KASSERT(global_u < _p_graph->global_n());

        if (_p_graph->is_owned_global_node(global_u)) {
            // move local nodes right away
            const NodeID local_u = _p_graph->global_to_local_node(global_u);
            _p_graph->set_block(local_u, to);
        } else {
            const PEID owner = _p_graph->find_owner_of_global_node(global_u);
            sendbuf[owner].emplace_back(static_cast<NodeID>(global_u - _p_graph->offset_n(owner)), to);
        }
    }
    STOP_TIMER();

    mpi::sparse_alltoall<MoveMessage>(
        sendbuf,
        [&](const auto& recvbuf) {
            tbb::parallel_for<std::size_t>(0, recvbuf.size(), [&](const std::size_t i) {
                const auto [node, block] = recvbuf[i];
                _p_graph->set_block(node, block);
            });
        },
        _p_graph->communicator());
    STOP_TIMER();

    START_TIMER("Synchronize ghost node labels");
    graph::synchronize_ghost_node_block_ids(p_graph);
    p_graph.reinit_block_weights();
    STOP_TIMER();
}

tbb::concurrent_vector<NodeID> FMRefiner::find_seed_nodes() {
    SCOPED_TIMER("Select seed nodes");

    // compute indepentent set
    NoinitVector<double>                          score(_p_graph->n());
    std::uniform_real_distribution<>              dist(0.0, 1.0 * _p_graph->total_n());
    tbb::enumerable_thread_specific<std::mt19937> generator_ets;

    _p_graph->pfor_nodes([&](const NodeID u) {
        // score for ghost nodes is computed lazy
        if (!_p_graph->is_owned_node(u)) {
            score[u] = -1.0;
            return;
        }

        // check if u is a bordert node
        bool          is_border_node = false;
        const BlockID u_block        = _p_graph->block(u);

        for (const auto [e, v]: _p_graph->neighbors(u)) {
            if (_p_graph->block(v) != u_block) {
                is_border_node = true;
                break;
            }
        }

        if (is_border_node) {
            auto& generator = generator_ets.local();
            generator.seed(_iteration + _p_graph->local_to_global_node(u));
            score[u] = dist(generator);
        } else {
            score[u] = std::numeric_limits<double>::max();
        }
    });

    // find seed nodes
    tbb::concurrent_vector<NodeID> seed_nodes;
    _p_graph->pfor_nodes([&](const NodeID u) {
        if (score[u] == std::numeric_limits<double>::max()) {
            return; // not a border node
        }

        bool is_seed_node = true;
        for (const auto [e, v]: _p_graph->neighbors(u)) {
            if (score[v] < 0) { // ghost node, compute score lazy
                auto& generator = generator_ets.local();
                generator.seed(_iteration + _p_graph->local_to_global_node(v));
                score[v] = dist(generator);
            }

            if (score[v] < score[u]) {
                is_seed_node = false;
                break;
            }
        }

        if (is_seed_node) {
            seed_nodes.push_back(u);
        }
    });

    DBG << "Selected " << seed_nodes.size() << " seed nodes";

    return seed_nodes;
}
} // namespace kaminpar::dist
