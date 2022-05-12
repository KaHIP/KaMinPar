/*******************************************************************************
 * @file:   distributed_metrics.cc
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Partition metrics for distributed graphs.
 ******************************************************************************/
#include "dkaminpar/utils/metrics.h"

#include <tbb/enumerable_thread_specific.h>

#include "context.h"
#include "datastructure/distributed_graph.h"
#include "dkaminpar/mpi_wrapper.h"

namespace dkaminpar::metrics {
GlobalEdgeWeight local_edge_cut(const DistributedPartitionedGraph& p_graph) {
    tbb::enumerable_thread_specific<GlobalEdgeWeight> cut_ets;

    p_graph.pfor_nodes_range([&](const auto r) {
        auto& cut = cut_ets.local();
        for (NodeID u = r.begin(); u < r.end(); ++u) {
            const BlockID u_block = p_graph.block(u);
            for (const auto [e, v]: p_graph.neighbors(u)) {
                if (u_block != p_graph.block(v)) {
                    cut += p_graph.edge_weight(e);
                }
            }
        }
    });

    return cut_ets.combine(std::plus{});
}

GlobalEdgeWeight edge_cut(const DistributedPartitionedGraph& p_graph) {
    const GlobalEdgeWeight global_edge_cut = mpi::allreduce(local_edge_cut(p_graph), MPI_SUM, p_graph.communicator());
    KASSERT(global_edge_cut % 2 == 0);
    return global_edge_cut / 2;
}

double imbalance(const DistributedPartitionedGraph& p_graph) {
    GlobalNodeWeight       total_local_node_weight = p_graph.total_node_weight();
    const GlobalNodeWeight global_total_node_weight =
        mpi::allreduce<GlobalNodeWeight>(total_local_node_weight, MPI_SUM, p_graph.communicator());

    const double perfect_block_weight = std::ceil(static_cast<double>(global_total_node_weight) / p_graph.k());
    double       max_imbalance        = 0.0;
    for (const BlockID b: p_graph.blocks()) {
        max_imbalance =
            std::max(max_imbalance, static_cast<double>(p_graph.block_weight(b)) / perfect_block_weight - 1.0);
    }

    return max_imbalance;
}

bool is_feasible(const DistributedPartitionedGraph& p_graph, const PartitionContext& p_ctx) {
    return imbalance(p_graph) < p_ctx.epsilon;
}

BlockID num_imbalanced_blocks(const DistributedPartitionedGraph& p_graph, const PartitionContext& p_ctx) {
    BlockID local_num_imbalanced_blocks = 0;

    for (const BlockID b: p_graph.blocks()) {
        if (p_graph.block_weight(b) > p_ctx.max_block_weight(b)) {
            ++local_num_imbalanced_blocks;
        }
    }

    return mpi::allreduce<BlockID>(local_num_imbalanced_blocks, MPI_SUM, p_graph.communicator());
}
} // namespace dkaminpar::metrics
