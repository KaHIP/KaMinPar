/*******************************************************************************
 * @file:   clustering_contraction.cc
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
#include "dkaminpar/coarsening/clustering_contraction.h"

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

std::pair<NoinitVector<GlobalNode>, NoinitVector<GlobalEdge>>
find_nonlocal_nodes_and_edges(const DistributedGraph& graph, const GlobalClustering& lnode_to_gcluster) {
    NoinitVector<NodeID> edge_position_buffer(graph.n() + 1);
    NoinitVector<NodeID> node_position_buffer(graph.n() + 1);
    edge_position_buffer.front() = 0;
    node_position_buffer.front() = 0;

    graph.pfor_nodes([&](const NodeID u) {
        const GlobalNodeID c_u = lnode_to_gcluster[u];

        if (graph.is_owned_global_node(c_u)) {
            edge_position_buffer[u + 1] = 0;
            node_position_buffer[u + 1] = 0;
        } else {
            edge_position_buffer[u + 1] = graph.degree(u);
            node_position_buffer[u + 1] = 1;
        }
    });

    parallel::prefix_sum(edge_position_buffer.begin(), edge_position_buffer.end(), edge_position_buffer.begin());
    parallel::prefix_sum(node_position_buffer.begin(), node_position_buffer.end(), node_position_buffer.begin());

    NoinitVector<GlobalEdge> nonlocal_edges(edge_position_buffer.back());
    NoinitVector<GlobalNode> nonlocal_nodes(node_position_buffer.back());

    graph.pfor_nodes([&](const NodeID u) {
        const GlobalNodeID gcluster = lnode_to_gcluster[u];

        if (!graph.is_owned_global_node(gcluster)) {
            // Node
            nonlocal_nodes[node_position_buffer[u]] = {.u = gcluster, .weight = graph.node_weight(u)};

            // Edge
            std::size_t pos = edge_position_buffer[u];
            for (const auto [e, v]: graph.neighbors(u)) {
                nonlocal_edges[pos] = {
                    .u      = gcluster,
                    .v      = lnode_to_gcluster[v],
                    .weight = graph.edge_weight(e),
                };
                ++pos;
            }
        }
    });

    return {std::move(nonlocal_nodes), std::move(nonlocal_edges)};
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
    std::exclusive_scan(distribution.begin(), distribution.end(), distribution.begin(), 0u);
    return distribution;
}

NoinitVector<NodeID> build_lnode_to_lcnode_mapping(
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

struct MigrationResult {
    NoinitVector<GlobalNode> local_nodes;
    NoinitVector<GlobalEdge> local_edges;

    // Can be re-used for mapping exchange ...
    std::vector<int> node_sendcounts;
    std::vector<int> node_sdispls;
    std::vector<int> node_recvcounts;
    std::vector<int> node_rdispls;
};

MigrationResult exchange_nonlocal_nodes_and_edges(
    const DistributedGraph& graph, const NoinitVector<GlobalNode>& nonlocal_nodes,
    const NoinitVector<GlobalEdge>& nonlocal_edges
) {
    const PEID size = mpi::get_comm_size(graph.communicator());

    parallel::vector_ets<EdgeID> num_edges_for_pe_ets(size);
    parallel::vector_ets<NodeID> num_nodes_for_pe_ets(size);

    tbb::parallel_invoke(
        [&] {
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
        },
        [&] {
            tbb::parallel_for<std::size_t>(0, nonlocal_nodes.size(), [&](const std::size_t i) {
                auto&      num_nodes_for_pe = num_nodes_for_pe_ets.local();
                const PEID pe               = graph.find_owner_of_global_node(nonlocal_nodes[i].u);
                ++num_nodes_for_pe[pe];
            });
        }
    );
    auto num_edges_for_pe = num_edges_for_pe_ets.combine(std::plus{});
    auto num_nodes_for_pe = num_nodes_for_pe_ets.combine(std::plus{});

    // Exchange edges
    NoinitVector<GlobalEdge> local_edges;
    std::vector<int>         local_edges_sendcounts(size);
    std::vector<int>         local_edges_sdispls(size);
    std::vector<int>         local_edges_recvcounts(size);
    std::vector<int>         local_edges_rdispls(size);

    std::copy(num_edges_for_pe.begin(), num_edges_for_pe.end(), local_edges_sendcounts.begin());
    std::exclusive_scan(local_edges_sendcounts.begin(), local_edges_sendcounts.end(), local_edges_sdispls.begin(), 0);
    MPI_Alltoall(
        local_edges_sendcounts.data(), 1, MPI_INT, local_edges_recvcounts.data(), 1, MPI_INT, graph.communicator()
    );
    std::exclusive_scan(local_edges_recvcounts.begin(), local_edges_recvcounts.end(), local_edges_rdispls.begin(), 0);

    local_edges.resize(local_edges_rdispls.back() + local_edges_recvcounts.back());
    MPI_Alltoallv(
        nonlocal_edges.data(), local_edges_sendcounts.data(), local_edges_sdispls.data(), mpi::type::get<GlobalEdge>(),
        local_edges.data(), local_edges_recvcounts.data(), local_edges_rdispls.data(), mpi::type::get<GlobalEdge>(),
        graph.communicator()
    );

    // Exchange nodes
    NoinitVector<GlobalNode> local_nodes;
    std::vector<int>         local_nodes_sendcounts(size);
    std::vector<int>         local_nodes_sdispls(size);
    std::vector<int>         local_nodes_recvcounts(size);
    std::vector<int>         local_nodes_rdispls(size);

    std::copy(num_nodes_for_pe.begin(), num_nodes_for_pe.end(), local_nodes_sendcounts.begin());
    std::exclusive_scan(local_nodes_sendcounts.begin(), local_nodes_sendcounts.end(), local_nodes_sdispls.begin(), 0);
    MPI_Alltoall(
        local_nodes_sendcounts.data(), 1, MPI_INT, local_nodes_recvcounts.data(), 1, MPI_INT, graph.communicator()
    );
    std::exclusive_scan(local_nodes_recvcounts.begin(), local_nodes_recvcounts.end(), local_nodes_rdispls.begin(), 0);

    local_nodes.resize(local_nodes_rdispls.back() + local_nodes_recvcounts.back());
    MPI_Alltoallv(
        nonlocal_nodes.data(), local_nodes_sendcounts.data(), local_nodes_sdispls.data(), mpi::type::get<GlobalNode>(),
        local_nodes.data(), local_nodes_recvcounts.data(), local_nodes_rdispls.data(), mpi::type::get<GlobalNode>(),
        graph.communicator()
    );

    return {
        .local_nodes     = std::move(local_nodes),
        .local_edges     = std::move(local_edges),
        .node_sendcounts = std::move(local_nodes_sendcounts),
        .node_sdispls    = std::move(local_nodes_sdispls),
        .node_recvcounts = std::move(local_nodes_recvcounts),
        .node_rdispls    = std::move(local_nodes_rdispls)};
}
} // namespace

ContractionResult contract_clustering(const DistributedGraph& graph, const GlobalClustering& lnode_to_gcluster) {
    SET_DEBUG(false);

    const PEID size = mpi::get_comm_size(graph.communicator());
    const PEID rank = mpi::get_comm_rank(graph.communicator());

    // Collect nodes and edges that must be migrated to another PE:
    // - nodes that are assigned to non-local clusters
    // - edges whose source is a node in a non-local cluster
    START_TIMER("Collect nonlocal nodes and edges");
    auto  nonlocal_elements = find_nonlocal_nodes_and_edges(graph, lnode_to_gcluster);
    auto& nonlocal_nodes    = nonlocal_elements.first;
    auto& nonlocal_edges    = nonlocal_elements.second;
    STOP_TIMER();

    // Deduplicate edges and nodes to reduce communication volume by aggregating node- and edge weights
    START_TIMER("Preprocess nonlocal nodes and edges");
    deduplicate_edge_list(nonlocal_edges);
    sort_node_list(nonlocal_nodes);
    STOP_TIMER();

    // Migrate those nodes and edges
    START_TIMER("Exchange nonlocal nodes and edges");
    auto  migration_result = exchange_nonlocal_nodes_and_edges(graph, nonlocal_nodes, nonlocal_edges);
    auto& local_nodes      = migration_result.local_nodes;
    auto& local_edges      = migration_result.local_edges;
    STOP_TIMER();

    // Sort the received edges so that we can aggregate them later
    START_TIMER("Sort received edges");
    tbb::parallel_sort(local_edges.begin(), local_edges.end(), [&](const auto& lhs, const auto& rhs) {
        return lhs.u < rhs.u;
    });
    STOP_TIMER();

    // Map non-empty clusters belonging to this PE to a consecutive range of coarse node IDs:
    // ```
    // lnode_to_lcnode[local node ID] = local coarse node ID
    // ```
    START_TIMER("Build lnode_to_lcnode[]");
    auto lnode_to_lcnode = build_lnode_to_lcnode_mapping(graph, lnode_to_gcluster, local_nodes);
    STOP_TIMER();

    // Make cluster IDs start at 0
    START_TIMER("Build coarse node distribution");
    const NodeID c_n                 = lnode_to_lcnode.empty() ? 0 : lnode_to_lcnode.back() + 1;
    auto         c_node_distribution = build_distribution<GlobalNodeID>(c_n, graph.communicator());
    DBG << "Coarse node distribution: [" << c_node_distribution << "]";
    STOP_TIMER();

    // To construct the mapping[] array, we need to know the mapping of nodes that we migrated to another PE to coarse
    // node IDs: exchange these mappings here
    START_TIMER("Exchange node mapping for migrated nodes");
    struct NodeMapping {
        GlobalNodeID gcluster;
        GlobalNodeID gcnode;
    };

    NoinitVector<NodeMapping> local_nodes_mapping(local_nodes.size());
    NoinitVector<NodeID>      migrated_nodes(local_nodes_mapping.size());

    tbb::parallel_for<std::size_t>(0, local_nodes.size(), [&](const std::size_t i) {
        const GlobalNodeID gcluster = local_nodes[i].u;
        const NodeID       lcluster = static_cast<NodeID>(gcluster - graph.offset_n());
        const NodeID       lcnode   = lnode_to_lcnode[lcluster];

        local_nodes_mapping[i] = {
            .gcluster = gcluster,
            .gcnode   = lcnode + c_node_distribution[rank],
        };
        migrated_nodes[i] = lcnode;
    });

    NoinitVector<NodeMapping> local_nodes_mapping_rsps(nonlocal_nodes.size());
    MPI_Alltoallv(
        local_nodes_mapping.data(), migration_result.node_recvcounts.data(), migration_result.node_rdispls.data(),
        mpi::type::get<NodeMapping>(), local_nodes_mapping_rsps.data(), migration_result.node_sendcounts.data(),
        migration_result.node_sdispls.data(), mpi::type::get<NodeMapping>(), graph.communicator()
    );
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
            const NodeID       coarse_local  = lnode_to_lcnode[local];
            const GlobalNodeID coarse_global = c_node_distribution[rank] + coarse_local;
            my_mapping_responses[pe][i]      = coarse_global;
        });
    });

    auto their_mapping_responses = mpi::sparse_alltoall_get<GlobalNodeID>(my_mapping_responses, graph.communicator());
    STOP_TIMER();

    // Build the coarse ghost node mapping: coarse ghost nodes to coarse global nodes
    START_TIMER("Build mapping");
    std::exclusive_scan(
        next_index_for_pe.begin(), next_index_for_pe.end(), next_index_for_pe.begin(),
        parallel::Aligned<parallel::Atomic<NodeID>>(0),
        [](const auto& init, const auto& op) {
            return parallel::Aligned<parallel::Atomic<NodeID>>(init.value + op.value);
        }
    );

    const NodeID c_ghost_n = next_index_for_pe.back().value;

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

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, local_nodes_mapping_rsps.size()), [&](const auto& r) {
        auto& handle = nonlocal_gcluster_to_index_handle_ets.local();
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
            const auto& [local_cluster, global_coarse_node] = local_nodes_mapping_rsps[i];
            handle.insert(local_cluster + 1, graph.global_n() + global_coarse_node + 1);
        }
    });

    // Build a mapping array from fine nodes to coarse nodes
    NoinitVector<GlobalNodeID> lnode_to_gcnode(graph.n());
    graph.pfor_nodes([&](const NodeID u) {
        const GlobalNodeID cluster = lnode_to_gcluster[u];

        if (graph.is_owned_global_node(cluster)) {
            lnode_to_gcnode[u] = lnode_to_lcnode[cluster - graph.offset_n()] + c_node_distribution[rank];
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
    STOP_TIMER();

    //
    // Sort local nodes by their cluster ID
    //
    localize_global_edge_list(local_edges, graph.offset_n(), lnode_to_lcnode);

    START_TIMER("Bucket sort nodes by cluster");
    auto  bucket_sort_result      = build_node_buckets(graph, lnode_to_lcnode, c_n, local_edges, lnode_to_gcluster);
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
                            const NodeID lcnode   = lnode_to_lcnode[lcluster];
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
                        for (std::size_t index = u - graph.n(); local_edges[index].u == lcu; ++index) {
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
        const NodeID c_u = lnode_to_lcnode[local_nodes[i].u - graph.offset_n()];
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
            .nodes      = std::move(migrated_nodes),
            .sendcounts = std::move(migration_result.node_recvcounts),
            .sdispls    = std::move(migration_result.node_rdispls),
            .recvcounts = std::move(migration_result.node_sendcounts),
            .rdispls    = std::move(migration_result.node_sdispls),
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
