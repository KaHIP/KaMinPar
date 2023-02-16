/*******************************************************************************
 * @file:   cluster_contraction.cc
 * @author: Daniel Seemaier
 * @date:   06.02.2023
 * @brief:  Graph contraction for arbitrary clusterings.
 *
 * In this file, we use the following naming sheme for node and cluster IDs:
 * - {g,l}[c]{node,cluster}
 *    ^ global or local ID
 *         ^ ID in [c]oarse graph or in fine graph
 *            ^ node or cluster ID
 ******************************************************************************/
#include "dkaminpar/coarsening/contraction/cluster_contraction.h"

#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>
#include <tbb/parallel_sort.h>

#include "dkaminpar/graphutils/communication.h"

#include "common/datastructures/rating_map.h"
#include "common/datastructures/ts_navigable_linked_list.h"
#include "common/noinit_vector.h"
#include "common/parallel/algorithm.h"
#include "common/parallel/aligned_element.h"
#include "common/parallel/vector_ets.h"
#include "common/timer.h"

namespace kaminpar::dist {
namespace {
struct GlobalEdge {
    GlobalNodeID u;
    GlobalNodeID v;
    EdgeWeight   weight;
};

struct GlobalNode {
    GlobalNodeID u;
    NodeWeight   weight;
};

NoinitVector<GlobalNode> find_nonlocal_nodes(const DistributedGraph& graph, const GlobalClustering& lnode_to_gcluster) {
    NoinitVector<NodeID> node_position_buffer(graph.n() + 1);
    node_position_buffer.front() = 0;
    graph.pfor_nodes([&](const NodeID lnode) {
        const GlobalNodeID gcluster     = lnode_to_gcluster[lnode];
        node_position_buffer[lnode + 1] = !graph.is_owned_global_node(gcluster);
    });
    parallel::prefix_sum(node_position_buffer.begin(), node_position_buffer.end(), node_position_buffer.begin());

    NoinitVector<GlobalNode> nonlocal_nodes(node_position_buffer.back());
    graph.pfor_nodes([&](const NodeID lnode) {
        const GlobalNodeID gcluster = lnode_to_gcluster[lnode];
        if (!graph.is_owned_global_node(gcluster)) {
            nonlocal_nodes[node_position_buffer[lnode]] = {.u = gcluster, .weight = graph.node_weight(lnode)};
        }
    });

    return nonlocal_nodes;
}

NoinitVector<GlobalEdge> find_nonlocal_edges(const DistributedGraph& graph, const GlobalClustering& lnode_to_gcluster) {
    NoinitVector<NodeID> edge_position_buffer(graph.n() + 1);
    edge_position_buffer.front() = 0;

    graph.pfor_nodes([&](const NodeID lnode) {
        const GlobalNodeID gcluster     = lnode_to_gcluster[lnode];
        edge_position_buffer[lnode + 1] = graph.is_owned_global_node(gcluster) ? 0 : graph.degree(lnode);
    });
    parallel::prefix_sum(edge_position_buffer.begin(), edge_position_buffer.end(), edge_position_buffer.begin());

    NoinitVector<GlobalEdge> nonlocal_edges(edge_position_buffer.back());
    graph.pfor_nodes([&](const NodeID lnode_u) {
        const GlobalNodeID gcluster_u = lnode_to_gcluster[lnode_u];

        if (!graph.is_owned_global_node(gcluster_u)) {
            std::size_t pos = edge_position_buffer[lnode_u];
            for (const auto [e, lnode_v]: graph.neighbors(lnode_u)) {
                const GlobalNodeID gcluster_v = lnode_to_gcluster[lnode_v];
                nonlocal_edges[pos]           = {
                              .u      = gcluster_u,
                              .v      = gcluster_v,
                              .weight = graph.edge_weight(e),
                };
                ++pos;
            }
        }
    });

    return nonlocal_edges;
}

void deduplicate_edge_list(NoinitVector<GlobalEdge>& edges) {
    if (edges.empty()) {
        return;
    }

    // Primary sort by edge source = messages are sorted by destination PE
    // Secondary sort by edge target = duplicate edges are consecutive
    tbb::parallel_sort(edges.begin(), edges.end(), [&](const auto& lhs, const auto& rhs) {
        return lhs.u < rhs.u || (lhs.u == rhs.u && lhs.v < rhs.v);
    });

    // Mark the first edge in every block of duplicate edges
    NoinitVector<EdgeID> edge_position_buffer(edges.size());
    edge_position_buffer.front() = 0;
    tbb::parallel_for<std::size_t>(1, edges.size(), [&](const std::size_t i) {
        edge_position_buffer[i] = (edges[i].u != edges[i - 1].u || edges[i].v != edges[i - 1].v);
    });

    // Prefix sum to get the location of the deduplicated edge
    parallel::prefix_sum(edge_position_buffer.begin(), edge_position_buffer.end(), edge_position_buffer.begin());

    // Deduplicate edges in a separate buffer
    NoinitVector<GlobalEdge> tmp_nonlocal_edges(edge_position_buffer.back() + 1);
    tbb::parallel_for<std::size_t>(0, edge_position_buffer.back() + 1, [&](const std::size_t i) {
        tmp_nonlocal_edges[i].weight = 0;
    });
    tbb::parallel_for<std::size_t>(0, edges.size(), [&](const std::size_t i) {
        const std::size_t pos = edge_position_buffer[i];
        __atomic_store_n(&(tmp_nonlocal_edges[pos].u), edges[i].u, __ATOMIC_RELAXED);
        __atomic_store_n(&(tmp_nonlocal_edges[pos].v), edges[i].v, __ATOMIC_RELAXED);
        __atomic_fetch_add(&(tmp_nonlocal_edges[pos].weight), edges[i].weight, __ATOMIC_RELAXED);
    });
    std::swap(tmp_nonlocal_edges, edges);
}

void sort_node_list(NoinitVector<GlobalNode>& nodes) {
    tbb::parallel_sort(nodes.begin(), nodes.end(), [&](const GlobalNode& lhs, const GlobalNode& rhs) {
        return lhs.u < rhs.u;
    });
}

void update_ghost_node_weights(DistributedGraph& graph) {
    SCOPED_TIMER("Update ghost node weights");

    struct Message {
        NodeID     local_node;
        NodeWeight weight;
    };

    mpi::graph::sparse_alltoall_interface_to_pe<Message>(
        graph,
        [&](const NodeID u) -> Message {
            return {u, graph.node_weight(u)};
        },
        [&](const auto buffer, const PEID pe) {
            tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
                const auto& [local_node_on_other_pe, weight] = buffer[i];
                const NodeID local_node = graph.global_to_local_node(graph.offset_n(pe) + local_node_on_other_pe);
                graph.set_ghost_node_weight(local_node, weight);
            });
        }
    );
}

template <typename T>
scalable_vector<T> build_distribution(const T count, MPI_Comm comm) {
    const PEID         size = mpi::get_comm_size(comm);
    scalable_vector<T> distribution(size + 1);
    MPI_Allgather(&count, 1, mpi::type::get<NodeID>(), distribution.data(), 1, mpi::type::get<GlobalNodeID>(), comm);
    std::exclusive_scan(distribution.begin(), distribution.end(), distribution.begin(), static_cast<T>(0));
    return distribution;
}

template <typename T>
double compute_distribution_imbalance(const scalable_vector<T>& distribution) {
    T max = 0;
    for (std::size_t i = 0; i + 1 < distribution.size(); ++i) {
        max = std::max(max, distribution[i + 1] - distribution[i]);
    }
    return 1.0 * max / (1.0 * distribution.back() / (distribution.size() - 1));
}

NoinitVector<NodeID> build_lcluster_to_lcnode_mapping(
    const DistributedGraph& graph, const GlobalClustering& lnode_to_gcluster,
    const NoinitVector<GlobalNode>& local_nodes
) {
    NoinitVector<NodeID> lcluster_to_lcnode(graph.n());
    graph.pfor_nodes([&](const NodeID u) { lcluster_to_lcnode[u] = 0; });
    tbb::parallel_invoke(
        [&] {
            graph.pfor_nodes([&](const NodeID lnode) {
                const GlobalNodeID gcluster = lnode_to_gcluster[lnode];
                if (graph.is_owned_global_node(gcluster)) {
                    const NodeID lcluster = static_cast<NodeID>(gcluster - graph.offset_n());
                    __atomic_store_n(&lcluster_to_lcnode[lcluster], 1, __ATOMIC_RELAXED);
                }
            });
        },
        [&] {
            tbb::parallel_for<std::size_t>(0, local_nodes.size(), [&](const std::size_t i) {
                const GlobalNodeID gcluster = local_nodes[i].u;
                KASSERT(graph.is_owned_global_node(gcluster));
                const NodeID lcluster = static_cast<NodeID>(gcluster - graph.offset_n());
                __atomic_store_n(&lcluster_to_lcnode[lcluster], 1, __ATOMIC_RELAXED);
            });
        }
    );
    parallel::prefix_sum(lcluster_to_lcnode.begin(), lcluster_to_lcnode.end(), lcluster_to_lcnode.begin());
    tbb::parallel_for<std::size_t>(0, lcluster_to_lcnode.size(), [&](const std::size_t i) {
        lcluster_to_lcnode[i] -= 1;
    });

    return lcluster_to_lcnode;
}

void localize_global_edge_list(
    NoinitVector<GlobalEdge>& edges, const GlobalNodeID offset, const NoinitVector<NodeID>& lnode_to_lcnode
) {
    tbb::parallel_for<std::size_t>(0, edges.size(), [&](const std::size_t i) {
        const NodeID lcluster = static_cast<NodeID>(edges[i].u - offset);
        edges[i].u            = lnode_to_lcnode[lcluster];
    });
}

std::pair<NoinitVector<NodeID>, NoinitVector<NodeID>> build_node_buckets(
    const DistributedGraph& graph, const NoinitVector<NodeID>& lcluster_to_lcnode, const GlobalNodeID c_n,
    const NoinitVector<GlobalEdge>& local_edges, const GlobalClustering& lnode_to_gcluster
) {
    NoinitVector<NodeID> buckets_position_buffer(c_n + 1);
    tbb::parallel_for<NodeID>(0, c_n + 1, [&](const NodeID lcnode) { buckets_position_buffer[lcnode] = 0; });

    tbb::parallel_invoke(
        [&] {
            graph.pfor_nodes([&](const NodeID lnode) {
                const GlobalNodeID gcluster = lnode_to_gcluster[lnode];
                if (graph.is_owned_global_node(gcluster)) {
                    const NodeID lcluster = static_cast<NodeID>(gcluster - graph.offset_n());
                    const NodeID lcnode   = lcluster_to_lcnode[lcluster];
                    KASSERT(lcnode < buckets_position_buffer.size());
                    __atomic_fetch_add(&buckets_position_buffer[lcnode], 1, __ATOMIC_RELAXED);
                }
            });
        },
        [&] {
            tbb::parallel_for<std::size_t>(0, local_edges.size(), [&](const std::size_t i) {
                if (i == 0 || local_edges[i].u != local_edges[i - 1].u) {
                    __atomic_fetch_add(&buckets_position_buffer[local_edges[i].u], 1, __ATOMIC_RELAXED);
                }
            });
        }
    );

    parallel::prefix_sum(
        buckets_position_buffer.begin(), buckets_position_buffer.end(), buckets_position_buffer.begin()
    );

    NoinitVector<NodeID> buckets(buckets_position_buffer.empty() ? 0 : buckets_position_buffer.back());
    tbb::parallel_invoke(
        [&] {
            graph.pfor_nodes([&](const NodeID lnode) {
                const GlobalNodeID gcluster = lnode_to_gcluster[lnode];
                if (graph.is_owned_global_node(gcluster)) {
                    const NodeID      lcluster = static_cast<NodeID>(gcluster - graph.offset_n());
                    const NodeID      lcnode   = lcluster_to_lcnode[lcluster];
                    const std::size_t pos = __atomic_fetch_sub(&buckets_position_buffer[lcnode], 1, __ATOMIC_RELAXED);
                    buckets[pos - 1]      = lnode;
                }
            });
        },
        [&] {
            tbb::parallel_for<std::size_t>(0, local_edges.size(), [&](const std::size_t i) {
                if (i == 0 || local_edges[i].u != local_edges[i - 1].u) {
                    const NodeID      lcnode = local_edges[i].u;
                    const std::size_t pos = __atomic_fetch_sub(&buckets_position_buffer[lcnode], 1, __ATOMIC_RELAXED);
                    buckets[pos - 1]      = graph.n() + i;
                }
            });
        }
    );

    return {std::move(buckets_position_buffer), std::move(buckets)};
}

template <typename T>
struct MigrationResult {
    NoinitVector<T> elements;

    // Can be re-used for mapping exchange ...
    std::vector<int> sendcounts;
    std::vector<int> sdispls;
    std::vector<int> recvcounts;
    std::vector<int> rdispls;
};

template <typename Element, typename NumElementsForPEContainer>
MigrationResult<Element> migrate_elements(
    const NumElementsForPEContainer& num_elements_for_pe, const NoinitVector<Element>& elements, MPI_Comm comm
) {
    const PEID size = mpi::get_comm_size(comm);

    NoinitVector<Element> recvbuf;
    std::vector<int>      sendcounts(size);
    std::vector<int>      sdispls(size);
    std::vector<int>      recvcounts(size);
    std::vector<int>      rdispls(size);

    std::copy(num_elements_for_pe.begin(), num_elements_for_pe.end(), sendcounts.begin());
    std::exclusive_scan(sendcounts.begin(), sendcounts.end(), sdispls.begin(), 0);
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);
    std::exclusive_scan(recvcounts.begin(), recvcounts.end(), rdispls.begin(), 0);

    recvbuf.resize(rdispls.back() + recvcounts.back());
    MPI_Alltoallv(
        elements.data(), sendcounts.data(), sdispls.data(), mpi::type::get<Element>(), recvbuf.data(),
        recvcounts.data(), rdispls.data(), mpi::type::get<Element>(), comm
    );

    return {
        .elements   = std::move(recvbuf),
        .sendcounts = std::move(sendcounts),
        .sdispls    = std::move(sdispls),
        .recvcounts = std::move(recvcounts),
        .rdispls    = std::move(rdispls)};
}

MigrationResult<GlobalNode>
migrate_nodes(const DistributedGraph& graph, const NoinitVector<GlobalNode>& nonlocal_nodes) {
    const PEID size = mpi::get_comm_size(graph.communicator());

    parallel::vector_ets<NodeID> num_nodes_for_pe_ets(size);
    tbb::parallel_for<std::size_t>(0, nonlocal_nodes.size(), [&](const std::size_t i) {
        auto&      num_nodes_for_pe = num_nodes_for_pe_ets.local();
        const PEID pe               = graph.find_owner_of_global_node(nonlocal_nodes[i].u);
        ++num_nodes_for_pe[pe];
    });
    auto num_nodes_for_pe = num_nodes_for_pe_ets.combine(std::plus{});

    return migrate_elements<GlobalNode>(num_nodes_for_pe, nonlocal_nodes, graph.communicator());
}

MigrationResult<GlobalEdge>
migrate_edges(const DistributedGraph& graph, const NoinitVector<GlobalEdge>& nonlocal_edges) {
    const PEID size = mpi::get_comm_size(graph.communicator());

    parallel::vector_ets<EdgeID> num_edges_for_pe_ets(size);
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(0, nonlocal_edges.size()),
        [&](const auto& r) {
            auto& num_edges_for_pe = num_edges_for_pe_ets.local();
            PEID  current_pe       = 0;
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                const GlobalNodeID u = nonlocal_edges[i].u;
                while (u >= graph.node_distribution(current_pe + 1)) {
                    ++current_pe;
                }
                ++num_edges_for_pe[current_pe];
            }
        },
        tbb::static_partitioner{}
    );
    auto num_edges_for_pe = num_edges_for_pe_ets.combine(std::plus{});

    return migrate_elements<GlobalEdge>(num_edges_for_pe, nonlocal_edges, graph.communicator());
}

struct NodeMapping {
    GlobalNodeID gcluster;
    GlobalNodeID gcnode;
};

struct MigratedNodesMapping {
    NoinitVector<NodeMapping> my_nonlocal_to_gcnode;
    NoinitVector<NodeID>      their_req_to_lcnode;
};

MigratedNodesMapping exchange_migrated_nodes_mapping(
    const DistributedGraph& graph, const NoinitVector<GlobalNode>& nonlocal_nodes,
    const MigrationResult<GlobalNode>& local_nodes, const NoinitVector<NodeID>& lcluster_to_lcnode,
    const scalable_vector<GlobalNodeID>& c_node_distribution
) {
    const PEID rank = mpi::get_comm_rank(graph.communicator());

    NoinitVector<NodeMapping> their_nonlocal_to_gcnode(local_nodes.elements.size());
    NoinitVector<NodeID>      their_req_to_lcnode(their_nonlocal_to_gcnode.size());

    tbb::parallel_for<std::size_t>(0, local_nodes.elements.size(), [&](const std::size_t i) {
        const GlobalNodeID gcluster = local_nodes.elements[i].u;
        const NodeID       lcluster = static_cast<NodeID>(gcluster - graph.offset_n());
        const NodeID       lcnode   = lcluster_to_lcnode[lcluster];

        their_nonlocal_to_gcnode[i] = {
            .gcluster = gcluster,
            .gcnode   = lcnode + c_node_distribution[rank],
        };
        their_req_to_lcnode[i] = lcnode;
    });

    NoinitVector<NodeMapping> my_nonlocal_to_gcnode(nonlocal_nodes.size());
    MPI_Alltoallv(
        their_nonlocal_to_gcnode.data(), local_nodes.recvcounts.data(), local_nodes.rdispls.data(),
        mpi::type::get<NodeMapping>(), my_nonlocal_to_gcnode.data(), local_nodes.sendcounts.data(),
        local_nodes.sdispls.data(), mpi::type::get<NodeMapping>(), graph.communicator()
    );

    return {std::move(my_nonlocal_to_gcnode), std::move(their_req_to_lcnode)};
}

template <typename T>
scalable_vector<T> create_perfect_distribution_from_global_count(const T global_count, MPI_Comm comm) {
    const auto size = mpi::get_comm_size(comm);

    scalable_vector<T> distribution(size + 1);
    for (PEID pe = 0; pe < size; ++pe) {
        distribution[pe + 1] = math::compute_local_range<T>(global_count, size, pe).second;
    }

    return distribution;
}

GlobalNodeID remap_lcnode(
    const GlobalNodeID lcnode, const PEID current_owner,
    const scalable_vector<GlobalNodeID>& current_cnode_distribution,
    const scalable_vector<GlobalNodeID>& desired_cnode_distribution, const NoinitVector<GlobalNodeID>& pe_overload,
    const NoinitVector<GlobalNodeID>& pe_underload
) {
    const auto local_node_count =
        static_cast<NodeID>(desired_cnode_distribution[current_owner + 1] - desired_cnode_distribution[current_owner]);
    if (lcnode < local_node_count) {
        return desired_cnode_distribution[current_owner] + lcnode;
    } else {
        const GlobalNodeID position = pe_overload[current_owner] + lcnode - local_node_count;
        const PEID new_owner = static_cast<PEID>(math::find_in_distribution<GlobalNodeID>(position, pe_underload));
        return desired_cnode_distribution[new_owner] + current_cnode_distribution[new_owner + 1]
               - current_cnode_distribution[new_owner] + position - pe_underload[new_owner];
    }
}

void rebalance_cluster_placement(
    const DistributedGraph& graph, const scalable_vector<GlobalNodeID>& current_cnode_distribution,
    const NoinitVector<NodeID>& lcluster_to_lcnode, const NoinitVector<NodeMapping>& my_nonlocal_gnode_to_gcluster,
    GlobalClustering& lnode_to_gcluster
) {
    const PEID         size = mpi::get_comm_size(graph.communicator());
    const PEID         rank = mpi::get_comm_rank(graph.communicator());
    const GlobalNodeID c_n  = current_cnode_distribution.back();

    const auto desired_cnode_distribution =
        create_perfect_distribution_from_global_count<GlobalNodeID>(c_n, graph.communicator());

    NoinitVector<GlobalNodeID> pe_overload(size + 1);
    NoinitVector<GlobalNodeID> pe_underload(size + 1);
    pe_overload.front()  = 0;
    pe_underload.front() = 0;

    for (PEID pe = 0; pe < size; ++pe) {
        const auto [from, to]      = math::compute_local_range<GlobalNodeID>(c_n, size, pe);
        const NodeID expected_size = static_cast<NodeID>(to - from);
        const NodeID actual_size =
            static_cast<NodeID>(current_cnode_distribution[pe + 1] - current_cnode_distribution[pe]);

        if (expected_size > actual_size) {
            pe_underload[pe + 1] = expected_size - actual_size;
        } else {
            pe_overload[pe + 1] = actual_size - expected_size;
        }
    }
    parallel::prefix_sum(pe_overload.begin(), pe_overload.end(), pe_overload.begin());
    parallel::prefix_sum(pe_underload.begin(), pe_underload.end(), pe_underload.begin());

    growt::GlobalNodeIDMap<GlobalNodeID> nonlocal_gnode_to_gcluster_map(my_nonlocal_gnode_to_gcluster.size());
    tbb::enumerable_thread_specific<growt::GlobalNodeIDMap<GlobalNodeID>::handle_type>
        nonlocal_gcnode_to_gcluster_handle_ets([&] { return nonlocal_gnode_to_gcluster_map.get_handle(); });

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, my_nonlocal_gnode_to_gcluster.size()), [&](const auto& r) {
        auto& handle = nonlocal_gcnode_to_gcluster_handle_ets.local();
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
            const auto& [gnode, gcluster] = my_nonlocal_gnode_to_gcluster[i];
            handle.insert(gnode + 1, gcluster);
        }
    });

    graph.pfor_nodes_range([&](const auto& r) {
        auto& handle = nonlocal_gcnode_to_gcluster_handle_ets.local();

        for (NodeID lnode = r.begin(); lnode != r.end(); ++lnode) {
            const GlobalNodeID old_gcluster = lnode_to_gcluster[lnode];
            GlobalNodeID       old_gcnode   = 0;
            PEID               new_owner    = 0;

            if (graph.is_owned_global_node(old_gcluster)) {
                const NodeID old_lcluster = static_cast<NodeID>(old_gcluster - graph.offset_n());
                old_gcnode                = lcluster_to_lcnode[old_lcluster] + current_cnode_distribution[rank];
                new_owner                 = rank;
            } else {
                auto it = handle.find(lnode + graph.offset_n() + 1);
                KASSERT(it != handle.end());
                old_gcnode = (*it).second;
                new_owner = 0; // @todo
            }

            const GlobalNodeID new_gcnode = remap_lcnode(
                old_gcnode, rank, current_cnode_distribution, desired_cnode_distribution, pe_overload, pe_underload
            );
            const GlobalNodeID new_gcluster =
                graph.offset_n(new_owner) + new_gcnode - desired_cnode_distribution[new_owner];
            lnode_to_gcluster[lnode] = new_gcluster;
        }
    });
}
} // namespace

ContractionResult contract_clustering(const DistributedGraph& graph, GlobalClustering& lnode_to_gcluster) {
    SET_DEBUG(false);

    const PEID size = mpi::get_comm_size(graph.communicator());
    const PEID rank = mpi::get_comm_rank(graph.communicator());

    // Collect nodes and edges that must be migrated to another PE:
    // - nodes that are assigned to non-local clusters
    // - edges whose source is a node in a non-local cluster
    START_TIMER("Collect nonlocal nodes");
    auto nonlocal_nodes = find_nonlocal_nodes(graph, lnode_to_gcluster);
    STOP_TIMER();

    START_TIMER("Preprocess nonlocal nodes");
    sort_node_list(nonlocal_nodes);
    STOP_TIMER();

    // Migrate those nodes and edges
    START_TIMER("Migrate nonlocal nodes");
    auto  migration_result_nodes = migrate_nodes(graph, nonlocal_nodes);
    auto& local_nodes            = migration_result_nodes.elements;
    STOP_TIMER();

    // Map non-empty clusters belonging to this PE to a consecutive range of coarse node IDs:
    // ```
    // lnode_to_lcnode[local node ID] = local coarse node ID
    // ```
    START_TIMER("Build lcluster_to_lcnode");
    auto lcluster_to_lcnode = build_lcluster_to_lcnode_mapping(graph, lnode_to_gcluster, local_nodes);
    STOP_TIMER();

    // Make cluster IDs start at 0
    START_TIMER("Build coarse node distribution");
    NodeID c_n                 = lcluster_to_lcnode.empty() ? 0 : lcluster_to_lcnode.back() + 1;
    auto   c_node_distribution = build_distribution<GlobalNodeID>(c_n, graph.communicator());
    DBG << "Coarse node distribution: [" << c_node_distribution << "]";
    STOP_TIMER();

    // To construct the mapping[] array, we need to know the mapping of nodes that we migrated to another PE to coarse
    // node IDs: exchange these mappings here
    START_TIMER("Exchange node mapping for migrated nodes");
    auto mapping_exchange_result = exchange_migrated_nodes_mapping(
        graph, nonlocal_nodes, migration_result_nodes, lcluster_to_lcnode, c_node_distribution
    );

    // Mapping from local nodes that belong to non-local clusters to coarse nodes:
    // .gcluster -- global cluster that belongs to another PE (but have at least one node in this cluster)
    // .gcnode -- the corresponding coarse node (global ID)
    auto& my_nonlocal_to_gcnode = mapping_exchange_result.my_nonlocal_to_gcnode;

    // Mapping from node migration messages that we received (i.e., messages that other PEs send us since they own nodes
    // that belong to some cluster owned by our PE) to the corresponding coarse node (local ID)
    // We don't need this information during contraction, but can use it to send the block assignment of coarse nodes to
    // other PEs during uncoarsening
    auto& their_req_to_lcnode = mapping_exchange_result.their_req_to_lcnode;
    STOP_TIMER();

    // If the "natural" assignment of coarse nodes to PEs has too much imbalance, we remap the cluster IDs to achieve
    // perfect coarse node balance
    if (compute_distribution_imbalance(c_node_distribution) > 1.03) { // @todo make this configurable
        rebalance_cluster_placement(
            graph, c_node_distribution, lcluster_to_lcnode, my_nonlocal_to_gcnode, lnode_to_gcluster
        );
        return contract_clustering(graph, lnode_to_gcluster);
    }

    START_TIMER("Collect nonlocal edges");
    auto nonlocal_edges = find_nonlocal_edges(graph, lnode_to_gcluster);
    STOP_TIMER();

    START_TIMER("Preprocess nonlocal edges");
    deduplicate_edge_list(nonlocal_edges);
    STOP_TIMER();

    START_TIMER("Migrate nonlocal edges");
    auto  migration_result_edges = migrate_edges(graph, nonlocal_edges);
    auto& local_edges            = migration_result_edges.elements;
    STOP_TIMER();

    START_TIMER("Sort migrated local edges");
    tbb::parallel_sort(local_edges.begin(), local_edges.end(), [&](const auto& lhs, const auto& rhs) {
        return lhs.u < rhs.u;
    });
    STOP_TIMER();

    // Next, exchange the mapping of ghost nodes to coarse nodes
    START_TIMER("Communicate mapping for ghost nodes");
    using NonlocalClusterMap = growt::GlobalNodeIDMap<GlobalNodeID>;
    NonlocalClusterMap nonlocal_gcluster_to_index(0);

    tbb::enumerable_thread_specific<NonlocalClusterMap::handle_type> nonlocal_gcluster_to_index_handle_ets([&] {
        return nonlocal_gcluster_to_index.get_handle();
    });

    std::vector<parallel::Aligned<parallel::Atomic<NodeID>>> next_index_for_pe(size + 1);

    auto request_nonlocal_mapping = [&](const GlobalNodeID cluster) {
        auto& handle          = nonlocal_gcluster_to_index_handle_ets.local();
        const auto [it, mine] = handle.insert(cluster + 1, 1); // dummy value
        if (mine) {
            const PEID owner = graph.find_owner_of_global_node(cluster);
            handle.update(cluster + 1, [&](auto& lhs) { return lhs = ++(next_index_for_pe[owner].value); });
        }
    };

    tbb::parallel_invoke(
        [&] {
            graph.pfor_nodes([&](const NodeID u) {
                const GlobalNodeID gcluster_u = lnode_to_gcluster[u];
                if (!graph.is_owned_global_node(gcluster_u)) {
                    return;
                }

                for (const auto [e, v]: graph.neighbors(u)) {
                    const GlobalNodeID gcluster_v = lnode_to_gcluster[v];
                    if (!graph.is_owned_global_node(gcluster_v)) {
                        request_nonlocal_mapping(gcluster_v);
                    }
                }
            });
        },
        [&] {
            tbb::parallel_for<std::size_t>(0, local_edges.size(), [&](const std::size_t i) {
                const GlobalNodeID gcluster_v = local_edges[i].v;
                if (!graph.is_owned_global_node(gcluster_v)) {
                    request_nonlocal_mapping(gcluster_v);
                }
            });
        }
    );

    std::vector<scalable_vector<GlobalNodeID>> my_mapping_requests(size);
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
        my_mapping_requests[pe].resize(next_index_for_pe[pe].value);
    });
    static std::atomic_size_t counter;
    counter = 0;

#pragma omp parallel
    {
        auto& handle = nonlocal_gcluster_to_index_handle_ets.local();

        const std::size_t capacity  = handle.capacity();
        std::size_t       cur_block = counter.fetch_add(4096);

        while (cur_block < capacity) {
            auto it = handle.range(cur_block, cur_block + 4096);
            for (; it != handle.range_end(); ++it) {
                const GlobalNodeID cluster        = (*it).first - 1;
                const PEID         owner          = graph.find_owner_of_global_node(cluster);
                const std::size_t  index          = (*it).second - 1;
                my_mapping_requests[owner][index] = cluster;
            }
            cur_block = counter.fetch_add(4096);
        }
    }

    auto their_mapping_requests = mpi::sparse_alltoall_get<GlobalNodeID>(my_mapping_requests, graph.communicator());

    std::vector<scalable_vector<GlobalNodeID>> my_mapping_responses(size);
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
        my_mapping_responses[pe].resize(their_mapping_requests[pe].size());

        tbb::parallel_for<std::size_t>(0, their_mapping_requests[pe].size(), [&](const std::size_t i) {
            const GlobalNodeID global        = their_mapping_requests[pe][i];
            const NodeID       local         = static_cast<NodeID>(global - graph.offset_n());
            const NodeID       coarse_local  = lcluster_to_lcnode[local];
            const GlobalNodeID coarse_global = c_node_distribution[rank] + coarse_local;
            my_mapping_responses[pe][i]      = coarse_global;
        });
    });

    auto their_mapping_responses = mpi::sparse_alltoall_get<GlobalNodeID>(my_mapping_responses, graph.communicator());
    STOP_TIMER();

    // Build the coarse ghost node mapping: coarse ghost nodes to coarse global nodes
    START_TIMER("Build mapping");
    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, my_nonlocal_to_gcnode.size()), [&](const auto& r) {
        auto& handle = nonlocal_gcluster_to_index_handle_ets.local();
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
            const auto& [local_cluster, global_coarse_node] = my_nonlocal_to_gcnode[i];
            handle.insert(local_cluster + 1, graph.global_n() + global_coarse_node + 1);
        }
    });

    // Build a mapping array from fine nodes to coarse nodes
    NoinitVector<GlobalNodeID> lnode_to_gcnode(graph.n());
    graph.pfor_nodes([&](const NodeID u) {
        const GlobalNodeID cluster = lnode_to_gcluster[u];

        if (graph.is_owned_global_node(cluster)) {
            lnode_to_gcnode[u] = lcluster_to_lcnode[cluster - graph.offset_n()] + c_node_distribution[rank];
        } else {
            auto& handle = nonlocal_gcluster_to_index_handle_ets.local();
            auto  it     = handle.find(cluster + 1);
            KASSERT(it != handle.end());

            const std::size_t index = (*it).second - 1;
            if (index < graph.global_n()) {
                const PEID owner   = graph.find_owner_of_global_node(cluster);
                lnode_to_gcnode[u] = their_mapping_responses[owner][index];
            } else {
                lnode_to_gcnode[u] = static_cast<GlobalNodeID>(index - graph.global_n());
            }
        }

        KASSERT(lnode_to_gcnode[u] < c_node_distribution.back());
    });

    // Build mapping for ghost nodes
    std::exclusive_scan(
        next_index_for_pe.begin(), next_index_for_pe.end(), next_index_for_pe.begin(),
        parallel::Aligned<parallel::Atomic<NodeID>>(0),
        [](const auto& init, const auto& op) {
            return parallel::Aligned<parallel::Atomic<NodeID>>(init.value + op.value);
        }
    );

    const NodeID                  c_ghost_n = next_index_for_pe.back().value;
    growt::StaticGhostNodeMapping c_global_to_ghost(c_ghost_n);
    scalable_vector<GlobalNodeID> c_ghost_to_global(c_ghost_n);
    scalable_vector<PEID>         c_ghost_owner(c_ghost_n);

    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
        for (std::size_t i = 0; i < my_mapping_requests[pe].size(); ++i) {
            const GlobalNodeID global = their_mapping_responses[pe][i];
            const NodeID       local  = next_index_for_pe[pe].value + i;
            c_global_to_ghost.insert(global + 1, c_n + local);
            c_ghost_to_global[local] = global;
            c_ghost_owner[local]     = pe;
        }
    });
    STOP_TIMER();

    //
    // Sort local nodes by their cluster ID
    //
    localize_global_edge_list(local_edges, graph.offset_n(), lcluster_to_lcnode);

    START_TIMER("Bucket sort nodes by cluster");
    auto  bucket_sort_result      = build_node_buckets(graph, lcluster_to_lcnode, c_n, local_edges, lnode_to_gcluster);
    auto& buckets_position_buffer = bucket_sort_result.first;
    auto& buckets                 = bucket_sort_result.second;
    STOP_TIMER();

    //
    // Construct the coarse edges
    //
    START_TIMER("Allocation");
    scalable_vector<EdgeID>     c_nodes(c_n + 1);
    scalable_vector<NodeWeight> c_node_weights(c_n + c_ghost_n);

    tbb::enumerable_thread_specific<RatingMap<EdgeWeight, NodeID>> collector_ets([&] {
        return RatingMap<EdgeWeight, NodeID>(c_n + c_ghost_n);
    });

    struct LocalEdge {
        NodeID     node;
        EdgeWeight weight;
    };

    NavigableLinkedList<NodeID, LocalEdge, scalable_vector> edge_buffer_ets;
    STOP_TIMER();

    START_TIMER("Construct edges");
    tbb::parallel_for(tbb::blocked_range<NodeID>(0, c_n), [&](const auto& r) {
        auto& collector   = collector_ets.local();
        auto& edge_buffer = edge_buffer_ets.local();

        for (NodeID lcu = r.begin(); lcu != r.end(); ++lcu) {
            edge_buffer.mark(lcu);

            const std::size_t first_pos = buckets_position_buffer[lcu];
            const std::size_t last_pos  = buckets_position_buffer[lcu + 1];

            auto collect_edges = [&](auto& map) {
                NodeWeight c_u_weight = 0;

                for (std::size_t i = first_pos; i < last_pos; ++i) {
                    const NodeID u = buckets[i];

                    auto handle_edge_to_gcluster = [&](const EdgeWeight weight, const GlobalNodeID gcluster) {
                        if (graph.is_owned_global_node(gcluster)) {
                            const NodeID lcluster = static_cast<NodeID>(gcluster - graph.offset_n());
                            const NodeID lcnode   = lcluster_to_lcnode[lcluster];
                            if (lcnode != lcu) {
                                map[lcnode] += weight;
                            }
                        } else {
                            auto& handle = nonlocal_gcluster_to_index_handle_ets.local();
                            auto  it     = handle.find(gcluster + 1);
                            KASSERT(it != handle.end() && (*it).second - 1 < graph.global_n());

                            const std::size_t  index     = (*it).second - 1;
                            const PEID         owner     = graph.find_owner_of_global_node(gcluster);
                            const GlobalNodeID gcnode    = their_mapping_responses[owner][index];
                            auto               lcnode_it = c_global_to_ghost.find(gcnode + 1);
                            const NodeID       lcnode    = (*lcnode_it).second;

                            if (lcnode != lcu) {
                                map[lcnode] += weight;
                            }
                        }
                    };

                    auto handle_edge_to_lnode = [&](const EdgeWeight weight, const NodeID lnode) {
                        if (graph.is_owned_node(lnode)) {
                            const GlobalNodeID gcnode = lnode_to_gcnode[lnode];
                            const bool         is_local_gcnode =
                                (gcnode >= c_node_distribution[rank] && gcnode < c_node_distribution[rank + 1]);

                            if (is_local_gcnode) {
                                const NodeID lcnode = static_cast<NodeID>(gcnode - c_node_distribution[rank]);
                                if (lcu != lcnode) {
                                    map[lcnode] += weight;
                                }
                            } else {
                                auto lcnode_it = c_global_to_ghost.find(gcnode + 1);
                                KASSERT(lcnode_it != c_global_to_ghost.end());
                                const NodeID lcnode = (*lcnode_it).second;
                                if (lcu != lcnode) {
                                    map[lcnode] += weight;
                                }
                            }
                        } else {
                            handle_edge_to_gcluster(weight, lnode_to_gcluster[lnode]);
                        }
                    };

                    if (u < graph.n()) {
                        c_u_weight += graph.node_weight(u);
                        for (const auto [e, v]: graph.neighbors(u)) {
                            handle_edge_to_lnode(graph.edge_weight(e), v);
                        }
                    } else {
                        // Fix node weight later
                        for (std::size_t index = u - graph.n();
                             index < local_edges.size() && local_edges[index].u == lcu; ++index) {
                            handle_edge_to_gcluster(local_edges[index].weight, local_edges[index].v);
                        }
                    }
                }

                c_node_weights[lcu] = c_u_weight;
                c_nodes[lcu + 1]    = map.size();

                for (const auto [c_v, weight]: map.entries()) {
                    edge_buffer.push_back({c_v, weight});
                }
                map.clear();
            };

            EdgeID upper_bound_degree = 0;
            for (std::size_t i = first_pos; i < last_pos; ++i) {
                const NodeID u = buckets[i];
                if (u < graph.n()) {
                    upper_bound_degree += graph.degree(u);
                } else {
                    upper_bound_degree += c_ghost_n; //@todo min max degree
                }
            }
            collector.update_upper_bound_size(upper_bound_degree);
            collector.run_with_map(collect_edges, collect_edges);
        }
    });

    parallel::prefix_sum(c_nodes.begin(), c_nodes.end(), c_nodes.begin());
    STOP_TIMER();

    START_TIMER("Integrate node weights of migrated nodes");
    tbb::parallel_for<std::size_t>(0, local_nodes.size(), [&](const std::size_t i) {
        const NodeID c_u = lcluster_to_lcnode[local_nodes[i].u - graph.offset_n()];
        __atomic_fetch_add(&c_node_weights[c_u], local_nodes[i].weight, __ATOMIC_RELAXED);
    });
    STOP_TIMER();

    // Build edge distribution
    START_TIMER("Build coarse edge distribution");
    const EdgeID c_m                 = c_nodes.back();
    auto         c_edge_distribution = build_distribution<GlobalEdgeID>(c_m, graph.communicator());
    DBG << "Coarse edge distribution: [" << c_edge_distribution << "]";
    STOP_TIMER();

    START_TIMER("Allocation");
    scalable_vector<NodeID>     c_edges(c_m);
    scalable_vector<EdgeWeight> c_edge_weights(c_m);
    STOP_TIMER();

    // Finally, build coarse graph
    START_TIMER("Construct coarse graph");
    auto all_buffered_nodes = ts_navigable_list::combine<NodeID, LocalEdge, scalable_vector>(edge_buffer_ets);

    tbb::parallel_for<NodeID>(0, c_n, [&](const NodeID i) {
        const auto&  marker = all_buffered_nodes[i];
        const auto*  list   = marker.local_list;
        const NodeID c_u    = marker.key;

        const EdgeID c_u_degree         = c_nodes[c_u + 1] - c_nodes[c_u];
        const EdgeID first_target_index = c_nodes[c_u];
        const EdgeID first_source_index = marker.position;

        for (std::size_t j = 0; j < c_u_degree; ++j) {
            const auto to            = first_target_index + j;
            const auto [c_v, weight] = list->get(first_source_index + j);
            c_edges[to]              = c_v;
            c_edge_weights[to]       = weight;
        }
    });

    DistributedGraph c_graph(
        std::move(c_node_distribution), std::move(c_edge_distribution), std::move(c_nodes), std::move(c_edges),
        std::move(c_node_weights), std::move(c_edge_weights), std::move(c_ghost_owner), std::move(c_ghost_to_global),
        std::move(c_global_to_ghost), false, graph.communicator()
    );
    STOP_TIMER();

    START_TIMER("Synchronize ghost node weights");
    update_ghost_node_weights(c_graph);
    STOP_TIMER();

    return {
        std::move(c_graph),
        std::move(lnode_to_gcnode),
        {
            .nodes      = std::move(their_req_to_lcnode),
            .sendcounts = std::move(migration_result_nodes.recvcounts),
            .sdispls    = std::move(migration_result_nodes.rdispls),
            .recvcounts = std::move(migration_result_nodes.sendcounts),
            .rdispls    = std::move(migration_result_nodes.sdispls),
        },
    };
}

DistributedPartitionedGraph project_partition(
    const DistributedGraph& graph, DistributedPartitionedGraph p_c_graph, const NoinitVector<GlobalNodeID>& c_mapping,
    const MigratedNodes& migration
) {
    struct MigratedNodeBlock {
        GlobalNodeID gcnode;
        BlockID      block;
    };
    NoinitVector<MigratedNodeBlock> migrated_nodes_sendbuf(migration.sdispls.back() + migration.sendcounts.back());
    NoinitVector<MigratedNodeBlock> migrated_nodes_recvbuf(migration.rdispls.back() + migration.recvcounts.back());

    TIMED_SCOPE("Exchange migrated node blocks") {
        tbb::parallel_for<std::size_t>(0, migrated_nodes_sendbuf.size(), [&](const std::size_t i) {
            const NodeID       lcnode = migration.nodes[i];
            const BlockID      block  = p_c_graph.block(lcnode);
            const GlobalNodeID gcnode = p_c_graph.local_to_global_node(lcnode);
            migrated_nodes_sendbuf[i] = {.gcnode = gcnode, .block = block};
        });

        MPI_Alltoallv(
            migrated_nodes_sendbuf.data(), migration.sendcounts.data(), migration.sdispls.data(),
            mpi::type::get<MigratedNodeBlock>(), migrated_nodes_recvbuf.data(), migration.recvcounts.data(),
            migration.rdispls.data(), mpi::type::get<MigratedNodeBlock>(), graph.communicator()
        );
    };

    START_TIMER("Allocation");
    scalable_vector<BlockID> partition(graph.total_n());
    STOP_TIMER();

    TIMED_SCOPE("Building projected partition array") {
        growt::GlobalNodeIDMap<GlobalNodeID>                                               gcnode_to_block(0);
        tbb::enumerable_thread_specific<growt::GlobalNodeIDMap<GlobalNodeID>::handle_type> gcnode_to_block_handle_ets(
            [&] { return gcnode_to_block.get_handle(); }
        );
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0, migrated_nodes_recvbuf.size()), [&](const auto& r) {
            auto& gcnode_to_block_handle = gcnode_to_block_handle_ets.local();
            for (std::size_t i = r.begin(); i != r.end(); ++i) {
                const auto& migrated_node = migrated_nodes_recvbuf[i];
                gcnode_to_block_handle.insert(migrated_node.gcnode + 1, migrated_node.block);
            }
        });

        graph.pfor_nodes_range([&](const auto& r) {
            auto& gcnode_to_block_handle = gcnode_to_block_handle_ets.local();

            for (NodeID u = r.begin(); u != r.end(); ++u) {
                const GlobalNodeID gcnode = c_mapping[u];
                if (p_c_graph.is_owned_global_node(gcnode)) {
                    const NodeID lcnode = p_c_graph.global_to_local_node(gcnode);
                    partition[u]        = p_c_graph.block(lcnode);
                } else {
                    auto it = gcnode_to_block_handle.find(gcnode + 1);
                    KASSERT(it != gcnode_to_block_handle.end(), V(gcnode));
                    partition[u] = (*it).second;
                }
            }
        });
    };

    struct GhostNodeLabel {
        NodeID  local_node_on_sender;
        BlockID block;
    };

    TIMED_SCOPE("Exchanging ghost node labels") {
        mpi::graph::sparse_alltoall_interface_to_pe<GhostNodeLabel>(
            graph,
            [&](const NodeID lnode) -> GhostNodeLabel {
                return {lnode, partition[lnode]};
            },
            [&](const auto buffer, const PEID pe) {
                tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
                    const auto& [sender_lnode, block] = buffer[i];
                    const GlobalNodeID gnode          = graph.offset_n(pe) + sender_lnode;
                    const NodeID       lnode          = graph.global_to_local_node(gnode);
                    partition[lnode]                  = block;
                });
            }
        );
    };

    return {&graph, p_c_graph.k(), std::move(partition), p_c_graph.take_block_weights()};
}
} // namespace kaminpar::dist
