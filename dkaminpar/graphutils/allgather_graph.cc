/*******************************************************************************
 * @file:   allgather_graph.cc
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Allgather a distributed graph to each PE.
 ******************************************************************************/
#include "dkaminpar/graphutils/allgather_graph.h"

#include <mpi.h>

#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_graph_builder.h"
#include "dkaminpar/definitions.h"
#include "dkaminpar/graphutils/graph_synchronization.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/mpi/utils.h"
#include "dkaminpar/mpi/wrapper.h"

#include "kaminpar/datastructures/graph.h"
#include "kaminpar/metrics.h"

#include "common/datastructures/static_array.h"
#include "common/parallel/atomic.h"

namespace kaminpar::dist::graph {
SET_DEBUG(true);

shm::Graph replicate_everywhere(const DistributedGraph& graph) {
    KASSERT(graph.global_n() < std::numeric_limits<NodeID>::max(), "number of nodes exceeds int size", assert::always);
    KASSERT(graph.global_m() < std::numeric_limits<EdgeID>::max(), "number of edges exceeds int size", assert::always);
    MPI_Comm comm = graph.communicator();

    // copy edges array with global node IDs
    StaticArray<NodeID> remapped_edges(graph.m());
    graph.pfor_nodes([&](const NodeID u) {
        for (const auto [e, v]: graph.neighbors(u)) {
            remapped_edges[e] = graph.local_to_global_node(v);
        }
    });

    // gather graph
    StaticArray<shm::EdgeID> nodes(graph.global_n() + 1);
    StaticArray<shm::NodeID> edges(graph.global_m());

    const bool is_node_weighted = mpi::allreduce<std::uint8_t>(graph.is_node_weighted(), MPI_MAX, graph.communicator());
    const bool is_edge_weighted = mpi::allreduce<std::uint8_t>(graph.is_edge_weighted(), MPI_MAX, graph.communicator());

    StaticArray<shm::NodeWeight> node_weights(is_node_weighted * graph.global_n());
    StaticArray<shm::EdgeWeight> edge_weights(is_edge_weighted * graph.global_m());

    auto nodes_recvcounts = mpi::build_distribution_recvcounts(graph.node_distribution());
    auto nodes_displs     = mpi::build_distribution_displs(graph.node_distribution());
    auto edges_recvcounts = mpi::build_distribution_recvcounts(graph.edge_distribution());
    auto edges_displs     = mpi::build_distribution_displs(graph.edge_distribution());

    mpi::allgatherv(
        graph.raw_nodes().data(), graph.n(), nodes.data(), nodes_recvcounts.data(), nodes_displs.data(), comm
    );
    mpi::allgatherv(
        remapped_edges.data(), remapped_edges.size(), edges.data(), edges_recvcounts.data(), edges_displs.data(), comm
    );
    if (is_node_weighted) {
        KASSERT((graph.is_node_weighted() || graph.n() == 0));
        if constexpr (std::is_same_v<shm::NodeWeight, NodeWeight>) {
            mpi::allgatherv(
                graph.raw_node_weights().data(), graph.n(), node_weights.data(), nodes_recvcounts.data(),
                nodes_displs.data(), comm
            );
        } else {
            StaticArray<NodeWeight> node_weights_buffer(graph.global_n());
            mpi::allgatherv(
                graph.raw_node_weights().data(), graph.n(), node_weights_buffer.data(), nodes_recvcounts.data(),
                nodes_displs.data(), comm
            );
            tbb::parallel_for<std::size_t>(0, node_weights_buffer.size(), [&](const std::size_t i) {
                node_weights[i] = node_weights_buffer[i];
            });
        }
    }
    if (is_edge_weighted) {
        KASSERT((graph.is_edge_weighted() || graph.m() == 0));
        if constexpr (std::is_same_v<shm::EdgeWeight, EdgeWeight>) {
            mpi::allgatherv(
                graph.raw_edge_weights().data(), graph.m(), edge_weights.data(), edges_recvcounts.data(),
                edges_displs.data(), comm
            );
        } else {
            StaticArray<EdgeWeight> edge_weights_buffer(graph.global_m());
            mpi::allgatherv(
                graph.raw_edge_weights().data(), graph.m(), edge_weights_buffer.data(), edges_recvcounts.data(),
                edges_displs.data(), comm
            );
            tbb::parallel_for<std::size_t>(0, edge_weights_buffer.size(), [&](const std::size_t i) {
                edge_weights[i] = edge_weights_buffer[i];
            });
        }
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

DistributedGraph replicate(const DistributedGraph& graph, const int num_replications) {
    const PEID size     = mpi::get_comm_size(graph.communicator());
    const PEID rank     = mpi::get_comm_rank(graph.communicator());
    const PEID new_size = size / num_replications;
    const PEID new_rank = rank / num_replications;

    // Communicator with relevant PEs
    MPI_Comm group;
    MPI_Comm_split(graph.communicator(), new_rank, rank, &group);
    const PEID group_size = num_replications;
    const PEID group_rank = rank - new_rank * num_replications;

    auto nodes_counts = mpi::build_counts_from_value<GlobalNodeID>(graph.n(), group);
    auto nodes_displs = mpi::build_displs_from_counts(nodes_counts);
    auto edges_counts = mpi::build_counts_from_value<GlobalEdgeID>(graph.m(), group);
    auto edges_displs = mpi::build_displs_from_counts(edges_counts);

    // Create edges array with global node IDs
    const GlobalEdgeID         my_tmp_global_edges_offset = edges_displs[group_rank];
    NoinitVector<GlobalNodeID> tmp_global_edges(edges_displs.back());
    graph.pfor_edges([&](const EdgeID e) {
        tmp_global_edges[my_tmp_global_edges_offset + e] = graph.local_to_global_node(graph.edge_target(e));
    });

    const bool is_node_weighted = mpi::allreduce<std::uint8_t>(graph.is_node_weighted(), MPI_MAX, graph.communicator());
    const bool is_edge_weighted = mpi::allreduce<std::uint8_t>(graph.is_edge_weighted(), MPI_MAX, graph.communicator());

    // Allocate memory for new graph
    scalable_vector<EdgeID>     nodes(nodes_displs.back() + 1);
    scalable_vector<NodeID>     edges(edges_displs.back());
    scalable_vector<EdgeWeight> edge_weights(is_edge_weighted ? edges_displs.back() : 0);

    // Exchange data -- except for node weights (must know the no. of ghost nodes to allocate the vector)
    mpi::allgatherv(graph.raw_nodes().data(), graph.n(), nodes.data(), nodes_counts.data(), nodes_displs.data(), group);
    MPI_Allgatherv(
        MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, tmp_global_edges.data(), edges_counts.data(), edges_displs.data(),
        mpi::type::get<GlobalEdgeID>(), group
    );
    if (is_edge_weighted) {
        KASSERT(graph.is_edge_weighted() || graph.m() == 0);
        mpi::allgatherv(
            graph.raw_edge_weights().data(), graph.m(), edge_weights.data(), edges_counts.data(), edges_displs.data(),
            group
        );
    }

    // Set nodes guard
    nodes.back() = edges_displs.back();

    // Offset received nodes arrays
    tbb::parallel_for<PEID>(0, group_size, [&](const PEID p) {
        const NodeID offset = edges_displs[p];
        tbb::parallel_for<NodeID>(nodes_displs[p], nodes_displs[p] + nodes_counts[p], [&](const NodeID u) {
            nodes[u] += offset;
        });
    });

    // Create new node and edges distributions
    scalable_vector<GlobalNodeID> node_distribution(new_size + 1);
    scalable_vector<GlobalEdgeID> edge_distribution(new_size + 1);
    tbb::parallel_for<PEID>(0, new_size, [&](const PEID pe) {
        node_distribution[pe + 1] = graph.node_distribution(group_size * (pe + 1));
        edge_distribution[pe + 1] = graph.edge_distribution(group_size * (pe + 1));
    });

    // Remap edges to local nodes
    const GlobalEdgeID n0 = graph.node_distribution(rank) - nodes_displs[group_rank];
    const GlobalEdgeID nf = n0 + nodes_displs.back();
    GhostNodeMapper    ghost_node_mapper(node_distribution, new_rank);

    tbb::parallel_for<EdgeID>(0, tmp_global_edges.size(), [&](const EdgeID e) {
        const GlobalNodeID v = tmp_global_edges[e];
        if (v >= n0 && v < nf) {
            edges[e] = static_cast<NodeID>(v - n0);
        } else {
            edges[e] = ghost_node_mapper.new_ghost_node(v);
        }
    });

    auto ghost_node_info = ghost_node_mapper.finalize();

    // Now that we know the number of ghost nodes: exchange node weights
    // The weights of ghost nodes are synchronized once the distributed graph data structure was built
    const NodeID                num_ghost_nodes = ghost_node_info.ghost_to_global.size();
    scalable_vector<NodeWeight> node_weights(is_node_weighted ? nodes_displs.back() + num_ghost_nodes : 0);
    if (is_node_weighted) {
        KASSERT(graph.is_node_weighted() || graph.n() == 0);
        mpi::allgatherv(
            graph.raw_node_weights().data(), graph.n(), node_weights.data(), nodes_counts.data(), nodes_displs.data(),
            group
        );
    }

    // Create new communicator and graph
    MPI_Comm new_comm;
    MPI_Comm_split(graph.communicator(), rank % num_replications, rank, &new_comm);
    KASSERT(mpi::get_comm_size(new_comm) == new_size);

    DistributedGraph new_graph(
        std::move(node_distribution), std::move(edge_distribution), std::move(nodes), std::move(edges),
        std::move(node_weights), std::move(edge_weights), std::move(ghost_node_info.ghost_owner),
        std::move(ghost_node_info.ghost_to_global), std::move(ghost_node_info.global_to_ghost), false, new_comm
    );

    // Fix weights of ghost nodes
    if (is_node_weighted) {
        synchronize_ghost_node_weights(new_graph);
    } else {
        tbb::parallel_for<NodeID>(new_graph.n(), new_graph.total_n(), [&](const NodeID u) {
            new_graph.set_ghost_node_weight(u, 1);
        });
    }

    KASSERT(debug::validate(new_graph), "", assert::heavy);

    MPI_Comm_free(&group);
    return new_graph;
}

DistributedPartitionedGraph
distribute_best_partition(const DistributedGraph& dist_graph, DistributedPartitionedGraph p_graph) {
    // Create group with one PE of each replication
    const PEID group_size       = mpi::get_comm_size(p_graph.communicator());
    const PEID group_rank       = mpi::get_comm_rank(p_graph.communicator());
    const PEID size             = mpi::get_comm_size(dist_graph.communicator());
    const PEID rank             = mpi::get_comm_rank(dist_graph.communicator());
    const PEID num_replications = size / group_size;

    MPI_Comm inter_group_comm;
    MPI_Comm_split(dist_graph.communicator(), group_rank, rank, &inter_group_comm);

    // Find best partition
    const GlobalEdgeWeight my_cut = metrics::edge_cut(p_graph);
    struct ReductionMessage {
        long cut;
        int  rank;
    };
    ReductionMessage best_cut_loc{my_cut, group_rank};
    MPI_Allreduce(MPI_IN_PLACE, &best_cut_loc, 1, MPI_LONG_INT, MPI_MINLOC, inter_group_comm);

    // Compute partition distribution for p_graph --> dist_graph
    NoinitVector<int> send_counts(num_replications);
    for (PEID pe = group_rank * num_replications; pe < (group_rank + 1) * num_replications; ++pe) {
        const PEID first_pe        = group_rank * num_replications;
        send_counts[pe - first_pe] = dist_graph.node_distribution(pe + 1) - dist_graph.node_distribution(pe);
    }
    NoinitVector<int> send_displs = mpi::build_displs_from_counts(send_counts);
    int               recv_count  = asserting_cast<int>(dist_graph.n());

    // Scatter best partition
    auto                     partition = p_graph.take_partition();
    scalable_vector<BlockID> new_partition(dist_graph.total_n());
    MPI_Scatterv(
        partition.data(), send_counts.data(), send_displs.data(), mpi::type::get<BlockID>(), new_partition.data(),
        recv_count, mpi::type::get<BlockID>(), best_cut_loc.rank, inter_group_comm
    );

    // Create partitioned dist_graph
    DistributedPartitionedGraph p_dist_graph(&dist_graph, p_graph.k(), std::move(new_partition));

    // Synchronize ghost node assignment
    synchronize_ghost_node_block_ids(p_dist_graph);

    return p_dist_graph;
}

DistributedPartitionedGraph
distribute_best_partition(const DistributedGraph& dist_graph, shm::PartitionedGraph shm_p_graph) {
    KASSERT(
        dist_graph.global_n() < static_cast<GlobalNodeID>(std::numeric_limits<NodeID>::max()),
        "partition size exceeds int size", assert::always
    );
    MPI_Comm comm = dist_graph.communicator();

    const int        rank    = mpi::get_comm_rank(comm);
    const EdgeWeight shm_cut = shm::metrics::edge_cut(shm_p_graph);

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
    MPI_Bcast(
        partition.data(), static_cast<int>(dist_graph.global_n()), mpi::type::get<shm::BlockID>(), global.rank, comm
    );

    // create distributed partition
    scalable_vector<BlockID> dist_partition(dist_graph.total_n());
    dist_graph.pfor_nodes(0, dist_graph.total_n(), [&](const NodeID u) {
        dist_partition[u] = partition[dist_graph.local_to_global_node(u)];
    });

    // create distributed partitioned graph
    return {&dist_graph, shm_p_graph.k(), std::move(dist_partition)};
}
} // namespace kaminpar::dist::graph
