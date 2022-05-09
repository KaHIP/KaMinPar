/*******************************************************************************
 * @file:   allgather_graph.cc
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Allgather a distributed graph to each PE.
 ******************************************************************************/
#include "dkaminpar/graphutils/allgather_graph.h"

#include "dkaminpar/mpi_wrapper.h"
#include "kaminpar/metrics.h"

namespace dkaminpar::graph {
SET_DEBUG(true);

shm::Graph allgather(const DistributedGraph& graph) {
    KASSERT(graph.global_n() < std::numeric_limits<NodeID>::max(), "number of nodes exceeds int size", assert::always);
    KASSERT(graph.global_m() < std::numeric_limits<EdgeID>::max(), "number of edges exceeds int size", assert::always);
    MPI_Comm comm = graph.communicator();

    // copy edges array with global node IDs
    shm::StaticArray<shm::NodeID> remapped_edges(graph.m());
    graph.pfor_nodes([&](const NodeID u) {
        for (const auto [e, v]: graph.neighbors(u)) {
            remapped_edges[e] = graph.local_to_global_node(v);
        }
    });

    // gather graph
    shm::StaticArray<shm::EdgeID> nodes(graph.global_n() + 1);
    shm::StaticArray<shm::NodeID> edges(graph.global_m());

    const bool is_node_weighted = mpi::allreduce<std::uint8_t>(graph.is_node_weighted(), MPI_MAX);
    const bool is_edge_weighted = mpi::allreduce<std::uint8_t>(graph.is_edge_weighted(), MPI_MAX);

    shm::StaticArray<shm::NodeWeight> node_weights(is_node_weighted * graph.global_n());
    shm::StaticArray<shm::EdgeWeight> edge_weights(is_edge_weighted * graph.global_m());

    auto nodes_recvcounts = mpi::build_distribution_recvcounts(graph.node_distribution());
    auto nodes_displs     = mpi::build_distribution_displs(graph.node_distribution());
    auto edges_recvcounts = mpi::build_distribution_recvcounts(graph.edge_distribution());
    auto edges_displs     = mpi::build_distribution_displs(graph.edge_distribution());

    mpi::allgatherv(
        graph.raw_nodes().data(), graph.n(), nodes.data(), nodes_recvcounts.data(), nodes_displs.data(), comm);
    if (is_node_weighted) {
        KASSERT((graph.is_node_weighted() || graph.n() == 0));
        mpi::allgatherv(
            graph.raw_node_weights().data(), graph.n(), node_weights.data(), nodes_recvcounts.data(),
            nodes_displs.data(), comm);
    }
    mpi::allgatherv(
        remapped_edges.data(), remapped_edges.size(), edges.data(), edges_recvcounts.data(), edges_displs.data(), comm);
    if (is_edge_weighted) {
        KASSERT((graph.is_edge_weighted() || graph.m() == 0));
        mpi::allgatherv(
            graph.raw_edge_weights().data(), graph.m(), edge_weights.data(), edges_recvcounts.data(),
            edges_displs.data(), comm);
    }
    nodes.back() = graph.global_m();

    // remap nodes array
    tbb::parallel_for(tbb::blocked_range<NodeID>(0, graph.global_n()), [&](const auto& r) {
        PEID pe = 0;
        for (NodeID u = r.begin(); u < r.end(); ++u) {
            while (u >= graph.node_distribution(pe + 1)) {
                KASSERT(pe < mpi::get_comm_size(comm));
                ++pe;
            }
            nodes[u] += graph.edge_distribution(pe);
        }
    });

    return {std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights)};
}

DistributedPartitionedGraph reduce_scatter(const DistributedGraph& dist_graph, shm::PartitionedGraph shm_p_graph) {
    KASSERT(
        dist_graph.global_n() < static_cast<GlobalNodeID>(std::numeric_limits<NodeID>::max()),
        "partition size exceeds int size", assert::always);
    MPI_Comm comm = dist_graph.communicator();

    const int             rank    = mpi::get_comm_rank(comm);
    const shm::EdgeWeight shm_cut = shm::metrics::edge_cut(shm_p_graph);

    // find PE with best partition
    struct ReductionMessage {
        long cut;
        int  rank;
    };
    ReductionMessage local{shm_cut, rank};
    ReductionMessage global{};
    MPI_Allreduce(&local, &global, 1, MPI_LONG_INT, MPI_MINLOC, comm);

    // broadcast best partition
    auto partition = shm_p_graph.take_partition();
    MPI_Bcast(partition.data(), static_cast<int>(dist_graph.global_n()), MPI_INT32_T, global.rank, comm);

    // compute block weights
    scalable_vector<shm::parallel::Atomic<BlockWeight>> block_weights(shm_p_graph.k());
    shm_p_graph.pfor_nodes([&](const shm::NodeID u) { block_weights[partition[u]] += shm_p_graph.node_weight(u); });

    // create distributed partition
    scalable_vector<Atomic<BlockID>> dist_partition(dist_graph.total_n());
    dist_graph.pfor_nodes(0, dist_graph.total_n(), [&](const NodeID u) {
        dist_partition[u] = partition[dist_graph.local_to_global_node(u)];
    });

    // create distributed partitioned graph
    return {&dist_graph, shm_p_graph.k(), std::move(dist_partition), std::move(block_weights)};
}
} // namespace dkaminpar::graph
