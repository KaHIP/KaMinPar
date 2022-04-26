/*******************************************************************************
 * @file:   distributed_balancer.cc
 *
 * @author: Daniel Seemaier
 * @date:   12.04.2022
 * @brief:  Distributed balancing refinement algorithm.
 ******************************************************************************/
#include "dkaminpar/refinement/distributed_balancer.h"

#include "dkaminpar/mpi_graph.h"
#include "dkaminpar/mpi_wrapper.h"
#include "kaminpar/utils/math.h"
#include "kaminpar/utils/random.h"
#include "kaminpar/utils/timer.h"

namespace dkaminpar {
DistributedBalancer::DistributedBalancer(const Context& ctx)
    : _ctx(ctx),
      _pq(ctx.partition.local_n(), ctx.partition.k),
      _pq_weight(ctx.partition.k),
      _marker(ctx.partition.local_n()) {}

void DistributedBalancer::initialize(const DistributedPartitionedGraph&) {}

void DistributedBalancer::balance(DistributedPartitionedGraph& p_graph, const PartitionContext& p_ctx) {
    const int rank = mpi::get_comm_rank(p_graph.communicator());

    _p_graph = &p_graph;
    _p_ctx   = &p_ctx;

    START_TIMER("Initialize PQ");
    init_pq();
    STOP_TIMER();

    while (true) {
        // pick best move candidates for each block
        START_TIMER("Pick candidates");
        auto candidates = pick_move_candidates();
        //print_candidates(candidates);
        STOP_TIMER();

        START_TIMER("Reudce");
        candidates = reduce_move_candidates(std::move(candidates));
        STOP_TIMER();

        //print_candidates(candidates);

        START_TIMER("Perform moves on root");
        if (rank == 0) {
            // move nodes that already have a target block
            perform_moves(candidates);

            // move nodes that do not have a target block
            BlockID cur = 0;
            for (auto& candidate: candidates) {
                auto& [node, from, to, weight, rel_gain] = candidate;

                if (from == to) {
                    // look for next block that can take node
                    while (cur == from || _p_graph->block_weight(cur) + weight > _p_ctx->max_block_weight(cur)) {
                        ++cur;
                        if (cur >= _p_ctx->k) {
                            cur = 0;
                        }
                    }

                    to = cur;
                    perform_move(candidate);
                }
            }
        }
        STOP_TIMER();

        // broadcast winners
        START_TIMER("Broadcast reduction result");
        const std::size_t num_winners = mpi::bcast(candidates.size(), 0, _p_graph->communicator());
        candidates.resize(num_winners);
        mpi::bcast(candidates.data(), num_winners, 0, _p_graph->communicator());
        STOP_TIMER();

        START_TIMER("Perform moves");
        if (rank != 0) {
            perform_moves(candidates);
        }
        STOP_TIMER();

        ASSERT([&] { graph::debug::validate_partition(*_p_graph); });

        if (num_winners == 0) {
            break;
        }
    }
}

void DistributedBalancer::print_candidates(const std::vector<MoveCandidate>& moves, const std::string &desc) const {
    std::stringstream ss;
    ss << desc << " [";
    for (const auto& [node, from, to, weight, rel_gain]: moves) {
        ss << "{node=" << node << ", from=" << from << ", to=" << to << ", weight=" << weight
           << ", rel_gain=" << rel_gain << "}";
    }
    ss << "]";
    DLOG << "candidates=" << ss.str();
}

void DistributedBalancer::print_overloads() const {
    for (const BlockID b: _p_graph->blocks()) {
        LOG << V(b) << V(block_overload(b));
    }
}

void DistributedBalancer::perform_moves(const std::vector<MoveCandidate>& moves) {
    for (const auto& move: moves) {
        perform_move(move);
    }
}

void DistributedBalancer::perform_move(const MoveCandidate& move) {
    const auto& [node, from, to, weight, rel_gain] = move;

    if (from == to) { // should only happen on root
        ASSERT(mpi::get_comm_rank(_p_graph->communicator()) == 0);
        return;
    }

    LOG << V(node) << V(from) << V(to) << V(weight) << V(rel_gain);

    if (_p_graph->contains_global_node(node)) {
        const NodeID u = _p_graph->global_to_local_node(node);

        if (_p_graph->graph().is_owned_global_node(node)) { // move node on this PE
            ASSERT(u < _p_graph->n());
            ASSERT(_pq.contains(u));

            _pq.remove(from, u);
            _pq_weight[from] -= weight;

            // activate neighbors
            for (const NodeID v: _p_graph->adjacent_nodes(u)) {
                if (!_p_graph->is_owned_node(v)) {
                    continue;
                }

                if (!_marker.get(v) && _p_graph->block(v) == from) {
                    add_to_pq(from, v);
                    _marker.set(v);
                }
            }
        }

        _p_graph->set_block(u, to);
    } else { // only update block weight
        _p_graph->set_block_weight(from, _p_graph->block_weight(from) - weight);
        _p_graph->set_block_weight(to, _p_graph->block_weight(to) + weight);
    }
}

auto DistributedBalancer::reduce_move_candidates(std::vector<MoveCandidate>&& candidates)
    -> std::vector<MoveCandidate> {
    const int size = mpi::get_comm_size(_p_graph->communicator());
    const int rank = mpi::get_comm_rank(_p_graph->communicator());
    ALWAYS_ASSERT(shm::math::is_power_of_2(size)) << "#PE must be a power of two";

    int active_size = size;
    while (active_size > 1) {
        if (rank >= active_size) {
            continue;
        }

        // false = receiver
        // true = sender
        const bool role = (rank >= active_size / 2);

        if (role) {
            const int dest = rank - active_size / 2;
            //print_candidates(candidates, "before send");
            mpi::send(candidates.data(), candidates.size(), dest, 0, _p_graph->communicator());
            return {};
        } else {
            const int                  src        = rank + active_size / 2;
            std::vector<MoveCandidate> tmp_buffer = mpi::probe_recv<MoveCandidate, std::vector<MoveCandidate>>(
                src, 0, MPI_STATUS_IGNORE, _p_graph->communicator());

            //print_candidates(tmp_buffer, "after recv");
            candidates = reduce_move_candidates(std::move(candidates), std::move(tmp_buffer));
        }

        active_size /= 2;
    }

    return candidates;
}

auto DistributedBalancer::reduce_move_candidates(std::vector<MoveCandidate>&& a, std::vector<MoveCandidate>&& b)
    -> std::vector<MoveCandidate> {
    std::vector<MoveCandidate> ans;

    // precondition: candidates are sorted by from block
    ASSERT([&] {
        for (std::size_t i = 1; i < a.size(); ++i) {
            ALWAYS_ASSERT(a[i].from >= a[i - 1].from);
        }
        for (std::size_t i = 1; i < b.size(); ++i) {
            ALWAYS_ASSERT(b[i].from >= b[i - 1].from);
        }
    });

    std::size_t i = 0; // index in a
    std::size_t j = 0; // index in b

    std::vector<NodeWeight> target_block_weight_delta(_p_ctx->k);

    for (i = 0, j = 0; i < a.size() && j < b.size();) {
        const BlockID from = std::min<BlockID>(a[i].from, b[j].from);

        // find region in `a` and `b` with nodes from `from`
        std::size_t i_end = i;
        std::size_t j_end = j;
        while (i_end < a.size() && a[i_end].from == from) {
            ++i_end;
        }
        while (j_end < b.size() && b[j_end].from == from) {
            ++j_end;
        }

        // pick best set of nodes
        const std::size_t num_in_a = i_end - i;
        const std::size_t num_in_b = j_end - j;
        const std::size_t num      = num_in_a + num_in_b;

        std::vector<MoveCandidate> candidates(num);
        std::copy(a.begin() + i, a.begin() + i_end, candidates.begin());
        std::copy(b.begin() + j, b.begin() + j_end, candidates.begin() + num_in_a);
        std::sort(candidates.begin(), candidates.end(), [&](const auto& lhs, const auto& rhs) {
            return lhs.rel_gain > rhs.rel_gain || (lhs.rel_gain == rhs.rel_gain && lhs.node > rhs.node);
        });
        // print_candidates(candidates);

        NodeWeight total_weight = 0;
        NodeID     added_to_ans = 0;
        for (NodeID candidate = 0; candidate < candidates.size(); ++candidate) {
            const BlockID    to     = candidates[candidate].to;
            const NodeWeight weight = candidates[candidate].weight;

            // only pick candidate if it does not overload the target block
            if (from != to
                && _p_graph->block_weight(to) + target_block_weight_delta[to] + weight > _p_ctx->max_block_weight(to)) {
                continue;
            }

            ans.push_back(candidates[candidate]);
            total_weight += weight;
            if (from != to) {
                target_block_weight_delta[to] += weight;
            }
            ++added_to_ans;

            // only pick candidates while we do not have enough weight to balance the block
            if (total_weight >= block_overload(from) || added_to_ans >= _ctx.refinement.balancing.num_nodes_per_block) {
                break;
            }
        }

        // move forward
        i = i_end;
        j = j_end;
    }

    // keep remaining moves
    while (i < a.size()) {
        ans.push_back(a[i++]);
    }
    while (j < b.size()) {
        ans.push_back(b[j++]);
    }

    return ans;
}

auto DistributedBalancer::pick_move_candidates() -> std::vector<MoveCandidate> {
    std::vector<MoveCandidate> candidates;

    for (const BlockID from: _p_graph->blocks()) {
        if (block_overload(from) == 0) {
            continue;
        }

        // fetch up to num_nodes_per_block move candidates from the PQ
        // but keep them in the PQ, since they might not get moved
        NodeID num = 0;
        for (num = 0; num < _ctx.refinement.balancing.num_nodes_per_block; ++num) {
            if (_pq.empty(from)) {
                break;
            }

            const NodeID     u             = _pq.peek_max_id(from);
            const double     relative_gain = _pq.peek_max_key(from);
            const NodeWeight u_weight      = _p_graph->node_weight(u);
            _pq.pop_max(from);
            _pq_weight[from] -= u_weight;

            auto [to, actual_relative_gain] = compute_gain(u, from);

            if (relative_gain == actual_relative_gain) {
                MoveCandidate candidate{_p_graph->local_to_global_node(u), from, to, u_weight, actual_relative_gain};
                candidates.push_back(candidate);
            } else {
                add_to_pq(from, u, u_weight, actual_relative_gain);
                --num; // retry
            }
        }

        for (NodeID rnum = 0; rnum < num; ++rnum) {
            ASSERT(candidates.size() > rnum);
            const auto& candidate = candidates[candidates.size() - rnum - 1];
            _pq.push(from, _p_graph->global_to_local_node(candidate.node), candidate.rel_gain);
            _pq_weight[from] += candidate.weight;
        }
    }

    return candidates;
}

void DistributedBalancer::init_pq() {
    SCOPED_TIMER("Initialize PQ");

    const BlockID k = _p_graph->k();

    tbb::enumerable_thread_specific<std::vector<shm::DynamicBinaryMinHeap<NodeID, double>>> local_pq_ets{[&] {
        return std::vector<shm::DynamicBinaryMinHeap<NodeID, double>>(k);
    }};

    tbb::enumerable_thread_specific<std::vector<NodeWeight>> local_pq_weight_ets{[&] {
        return std::vector<NodeWeight>(k);
    }};

    _marker.reset();

    // build thread-local PQs: one PQ for each thread and block, each PQ for block b has at most roughly
    // |overload[b]| weight
    START_TIMER("Thread-local");
    tbb::parallel_for(static_cast<NodeID>(0), _p_graph->n(), [&](const NodeID u) {
        auto& pq        = local_pq_ets.local();
        auto& pq_weight = local_pq_weight_ets.local();

        const BlockID     b        = _p_graph->block(u);
        const BlockWeight overload = block_overload(b);

        if (overload > 0) { // node in overloaded block
            const auto [max_gainer, rel_gain] = compute_gain(u, b);
            const bool need_more_nodes        = (pq_weight[b] < overload);
            if (need_more_nodes || pq[b].empty() || rel_gain > pq[b].peek_key()) {
                if (!need_more_nodes) {
                    const NodeWeight u_weight   = _p_graph->node_weight(u);
                    const NodeWeight min_weight = _p_graph->node_weight(pq[b].peek_id());
                    if (pq_weight[b] + u_weight - min_weight >= overload) {
                        pq[b].pop();
                    }
                }
                pq[b].push(u, rel_gain);
                _marker.set(u);
            }
        }
    });
    STOP_TIMER();

    // build global PQ: one PQ per block, block-level parallelism
    _pq.clear();

    START_TIMER("Merge thread-local PQs");
    tbb::parallel_for(static_cast<BlockID>(0), k, [&](const BlockID b) {
        _pq_weight[b] = 0;

        for (auto& pq: local_pq_ets) {
            for (const auto& [u, rel_gain]: pq[b].elements()) {
                add_to_pq(b, u, _p_graph->node_weight(u), rel_gain);
            }
        }
    });
    STOP_TIMER();
}

std::pair<BlockID, double> DistributedBalancer::compute_gain(const NodeID u, const BlockID u_block) const {
    const NodeWeight u_weight          = _p_graph->node_weight(u);
    BlockID          max_gainer        = u_block;
    EdgeWeight       max_external_gain = 0;
    EdgeWeight       internal_degree   = 0;

    auto action = [&](auto& map) {
        // compute external degree to each adjacent block that can take u without becoming overloaded
        for (const auto [e, v]: _p_graph->neighbors(u)) {
            const BlockID v_block = _p_graph->block(v);
            if (u_block != v_block && _p_graph->block_weight(v_block) + u_weight <= _p_ctx->max_block_weight(v_block)) {
                map[v_block] += _p_graph->edge_weight(e);
            } else if (u_block == v_block) {
                internal_degree += _p_graph->edge_weight(e);
            }
        }

        // select neighbor that maximizes gain
        auto& rand = shm::Randomize::instance();
        for (const auto [block, gain]: map.entries()) {
            if (gain > max_external_gain || (gain == max_external_gain && rand.random_bool())) {
                max_gainer        = block;
                max_external_gain = gain;
            }
        }
        map.clear();
    };

    auto& rating_map = _rating_map.local();
    rating_map.update_upper_bound_size(_p_graph->degree(u));
    rating_map.run_with_map(action, action);

    // compute absolute and relative gain based on internal degree / external gain
    const EdgeWeight gain          = max_external_gain - internal_degree;
    const double     relative_gain = compute_relative_gain(gain, u_weight);
    return {max_gainer, relative_gain};
}

BlockWeight DistributedBalancer::block_overload(const BlockID b) const {
    static_assert(
        std::numeric_limits<BlockWeight>::is_signed,
        "This must be changed when using an unsigned data type for block weights!");
    return std::max<BlockWeight>(0, _p_graph->block_weight(b) - _p_ctx->max_block_weight(b));
}

double DistributedBalancer::compute_relative_gain(const EdgeWeight absolute_gain, const NodeWeight weight) const {
    if (absolute_gain >= 0) {
        return absolute_gain * weight;
    } else {
        return 1.0 * absolute_gain / weight;
    }
}

bool DistributedBalancer::add_to_pq(const BlockID b, const NodeID u) {
    ASSERT(b == _p_graph->block(u));

    const auto [to, rel_gain] = compute_gain(u, b);
    return add_to_pq(b, u, _p_graph->node_weight(u), rel_gain);
}

bool DistributedBalancer::add_to_pq(const BlockID b, const NodeID u, const NodeWeight u_weight, const double rel_gain) {
    ASSERT(u_weight == _p_graph->node_weight(u));
    ASSERT(b == _p_graph->block(u));

    if (_pq_weight[b] < block_overload(b) || _pq.empty(b) || rel_gain > _pq.peek_min_key(b)) {
        _pq.push(b, u, rel_gain);
        _pq_weight[b] += u_weight;

        if (rel_gain > _pq.peek_min_key(b)) {
            const NodeID     min_node   = _pq.peek_min_id(b);
            const NodeWeight min_weight = _p_graph->node_weight(min_node);
            if (_pq_weight[b] - min_weight >= block_overload(b)) {
                _pq.pop_min(b);
                _pq_weight[b] -= min_weight;
            }
        }

        return true;
    }

    return false;
}
} // namespace dkaminpar
