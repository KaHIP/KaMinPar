#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/mpi/wrapper.h"

namespace dkaminpar::testing {
inline DistributedPartitionedGraph
make_partitioned_graph(const DistributedGraph& graph, const BlockID k, const std::vector<BlockID>& local_partition) {
    scalable_vector<parallel::Atomic<BlockID>> partition(graph.total_n());
    scalable_vector<BlockWeight>               local_block_weights(k);

    std::copy(local_partition.begin(), local_partition.end(), partition.begin());
    for (const NodeID u: graph.nodes()) {
        local_block_weights[partition[u]] += graph.node_weight(u);
    }

    scalable_vector<BlockWeight> global_block_weights_nonatomic(k);
    mpi::allreduce(local_block_weights.data(), global_block_weights_nonatomic.data(), k, MPI_SUM, MPI_COMM_WORLD);

    scalable_vector<parallel::Atomic<BlockWeight>> block_weights(k);
    std::copy(global_block_weights_nonatomic.begin(), global_block_weights_nonatomic.end(), block_weights.begin());

    struct NodeBlock {
        GlobalNodeID global_node;
        BlockID      block_weights;
    };

    mpi::graph::sparse_alltoall_interface_to_pe<NodeBlock>(
        graph,
        [&](const NodeID u) {
            return NodeBlock{graph.local_to_global_node(u), local_partition[u]};
        },
        [&](const auto& buffer) {
            for (const auto& [global_node, block]: buffer) {
                partition[graph.global_to_local_node(global_node)] = block;
            }
        });

    return {&graph, k, std::move(partition), std::move(block_weights)};
}

inline DistributedPartitionedGraph make_partitioned_graph_by_rank(const DistributedGraph& graph) {
    const PEID           rank = mpi::get_comm_rank(graph.communicator());
    const PEID           size = mpi::get_comm_size(graph.communicator());
    std::vector<BlockID> local_partition(graph.n(), rank);
    return make_partitioned_graph(graph, size, local_partition);
}
} // namespace dkaminpar::testing
