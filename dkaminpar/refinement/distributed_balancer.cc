/*******************************************************************************
 * @file:   distributed_balancer.cc
 *
 * @author: Daniel Seemaier
 * @date:   12.04.2022
 * @brief:  Distributed balancing refinement algorithm.
 ******************************************************************************/
#include "dkaminpar/refinement/distributed_balancer.h"

#include "kaminpar/utils/random.h"

namespace dkaminpar {
DistributedBalancer::DistributedBalancer(const Context& ctx)
    : _ctx(ctx),
      _pq(ctx.partition.local_n(), ctx.partition.k),
      _pq_weight(ctx.partition.k),
      _marker(ctx.partition.local_n()) {}

void DistributedBalancer::initialize(const DistributedPartitionedGraph&) {}

void DistributedBalancer::balance(DistributedPartitionedGraph& p_graph, const PartitionContext& p_ctx) {
    _p_graph = &p_graph;
    _p_ctx   = &p_ctx;

    init_pq();
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

    // build thread-local PQs: one PQ for each thread and block, each PQ for block b has at most roughly |overload[b]|
    // weight
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
