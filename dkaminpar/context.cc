/*******************************************************************************
 * @file:   context.cc
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Context struct for the distributed graph partitioner.
 ******************************************************************************/
#include "dkaminpar/context.h"

#include <unordered_map>

#include <tbb/parallel_for.h>

#include "dkaminpar/mpi/wrapper.h"

namespace kaminpar::dist {
using namespace std::string_literals;
//
// Functions for compact, parsable context output
//

void PartitionContext::setup(const DistributedGraph& graph) {
    _global_n = graph.global_n();
    _global_m = graph.global_m();
    _global_total_node_weight =
        mpi::allreduce<GlobalNodeWeight>(graph.total_node_weight(), MPI_SUM, graph.communicator());
    _local_n                = graph.n();
    _total_n                = graph.total_n();
    _local_m                = graph.m();
    _total_node_weight      = graph.total_node_weight();
    _global_max_node_weight = graph.global_max_node_weight();

    setup_perfectly_balanced_block_weights();
    setup_max_block_weights();
}

void PartitionContext::setup(const shm::Graph& graph) {
    _global_n                 = graph.n();
    _global_m                 = graph.m();
    _global_total_node_weight = graph.total_node_weight();
    _local_n                  = graph.n();
    _total_n                  = graph.n();
    _local_m                  = graph.m();
    _total_node_weight        = graph.total_node_weight();
    _global_max_node_weight   = graph.max_node_weight();

    setup_perfectly_balanced_block_weights();
    setup_max_block_weights();
}

void PartitionContext::setup_perfectly_balanced_block_weights() {
    _perfectly_balanced_block_weights.resize(k);

    const BlockWeight perfectly_balanced_block_weight = std::ceil(static_cast<double>(global_total_node_weight()) / k);
    tbb::parallel_for<BlockID>(0, k, [&](const BlockID b) {
        _perfectly_balanced_block_weights[b] = perfectly_balanced_block_weight;
    });
}

void PartitionContext::setup_max_block_weights() {
    _max_block_weights.resize(k);

    tbb::parallel_for<BlockID>(0, k, [&](const BlockID b) {
        const BlockWeight max_eps_weight =
            static_cast<BlockWeight>((1.0 + epsilon) * static_cast<double>(perfectly_balanced_block_weight(b)));
        const BlockWeight max_abs_weight = perfectly_balanced_block_weight(b) + _global_max_node_weight;

        // Only relax weight on coarse levels
        if (static_cast<GlobalNodeWeight>(_global_n) == _global_total_node_weight) {
            _max_block_weights[b] = max_eps_weight;
        } else {
            _max_block_weights[b] = std::max(max_eps_weight, max_abs_weight);
        }
    });
}

[[nodiscard]] bool
LabelPropagationCoarseningContext::should_merge_nonadjacent_clusters(const NodeID old_n, const NodeID new_n) const {
    return (1.0 - 1.0 * static_cast<double>(new_n) / static_cast<double>(old_n))
           <= merge_nonadjacent_clusters_threshold;
}

void LabelPropagationCoarseningContext::setup(const ParallelContext& parallel) {
    if (num_chunks == 0) {
        const std::size_t chunks =
            scale_chunks_with_threads ? total_num_chunks / parallel.num_threads : total_num_chunks;
        num_chunks = std::max<std::size_t>(8, chunks / parallel.num_mpis);
    }
}

void LabelPropagationRefinementContext::setup(const ParallelContext& parallel) {
    if (num_chunks == 0) {
        const std::size_t chunks =
            scale_chunks_with_threads ? total_num_chunks / parallel.num_threads : total_num_chunks;
        num_chunks = std::max<std::size_t>(8, chunks / parallel.num_mpis);
    }
}

void CoarseningContext::setup(const ParallelContext& parallel) {
    local_lp.setup(parallel);
    global_lp.setup(parallel);
}

void RefinementContext::setup(const ParallelContext& parallel) {
    lp.setup(parallel);
}

void Context::setup(const DistributedGraph& graph) {
    coarsening.setup(parallel);
    refinement.setup(parallel);
    partition.graph = GraphContext(graph, partition);
}
} // namespace kaminpar::dist
