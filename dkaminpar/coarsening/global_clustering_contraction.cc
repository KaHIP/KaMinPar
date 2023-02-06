/*******************************************************************************
 * @file:   global_clustering_contraction_redistribution.cc
 * @author: Daniel Seemaier
 * @date:   28.10.2021
 * @brief:  Shared-memory parallel contraction of global clustering without
 * any restrictions.
 ******************************************************************************/
#include "dkaminpar/coarsening/global_clustering_contraction.h"

#include <numeric>

#include <mpi.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/task_arena.h>

#include "dkaminpar/coarsening/contraction_helper.h"
#include "dkaminpar/datastructures/distributed_graph_builder.h"
#include "dkaminpar/datastructures/growt.h"
#include "dkaminpar/graphutils/communication.h"
#include "dkaminpar/mpi/wrapper.h"

#include "common/datastructures/rating_map.h"
#include "common/noinit_vector.h"
#include "common/parallel/atomic.h"
#include "common/parallel/loops.h"
#include "common/parallel/vector_ets.h"

namespace kaminpar::dist {
using namespace helper;

namespace {
/*!
 * Sparse all-to-all to exchange coarse node IDs of ghost nodes.
 * @param graph
 * @param label_mapping Current coarse node IDs, must be of size \code{graph.total_n()}, i.e., large enough to store
 * coarse node IDs of owned nodes and ghost nodes.
 */
template <typename LabelMapping>
void exchange_ghost_node_mapping(const DistributedGraph& graph, LabelMapping& label_mapping) {
    SCOPED_TIMER("Exchange ghost node mapping");

    struct Message {
        NodeID       local_node;
        GlobalNodeID coarse_global_node;
    };

    mpi::graph::sparse_alltoall_interface_to_pe<Message>(
        graph,
        [&](const NodeID u) -> Message {
            return {u, label_mapping[u]};
        },
        [&](const auto buffer, const PEID pe) {
            tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
                const auto& [local_node_on_pe, coarse_global_node] = buffer[i];
                const GlobalNodeID global_node                     = graph.offset_n(pe) + local_node_on_pe;
                const auto         local_node                      = graph.global_to_local_node(global_node);

                label_mapping[local_node] = coarse_global_node;
            });
        }
    );
}

using UsedClustersMap    = growt::GlobalNodeIDMap<GlobalNodeID>;
using UsedClustersVector = scalable_vector<NodeID>;

/*!
 * Given a graph with a mapping from nodes to clusters, finds the unique set of clusters that are used by the mapped
 * nodes. Each cluster is owned by some PE (determined by \c resolve_cluster_callback). For each PE, the function
 * returns a map and a vector of local cluster IDs used by the mapped nodes of this PE.
 *
 * @tparam ResolveClusterCallback
 * @param graph
 * @param clustering
 * @param resolve_cluster_callback Given a cluster ID, returns the owner PE (PEID) and the local node/cluster ID
 * (NodeID).
 * @return First component: for each PE \c p, a map mapping local cluster IDs on PE \c p used by mapped nodes on this
 * PE to entries in the second component; Second component: for each PE \c p, a vector containing all local cluster IDs
 * on PE \c p used by mapped nodes on this PE.
 */
template <typename ResolveClusterCallback, typename Clustering>
std::pair<std::vector<UsedClustersMap>, std::vector<UsedClustersVector>> find_used_cluster_ids_per_pe(
    const DistributedGraph& graph, const Clustering& clustering, ResolveClusterCallback&& resolve_cluster_callback
) {
    SCOPED_TIMER("find_used_cluster_ids_per_pe()");

    const auto size = mpi::get_comm_size(graph.communicator());

    // mark global node IDs that are used as cluster IDs
    std::vector<UsedClustersMap> used_clusters_map;
    used_clusters_map.reserve(size);
    for (PEID pe = 0; pe < size; ++pe) {
        used_clusters_map.emplace_back(0);
    }

    std::vector<parallel::Atomic<NodeID>> next_slot_for_pe(size);

    tbb::enumerable_thread_specific<std::vector<UsedClustersMap::handle_type>> used_clusters_map_handles([&] {
        std::vector<UsedClustersMap::handle_type> handles;
        handles.reserve(size);
        for (PEID pe = 0; pe < size; ++pe) {
            handles.push_back(used_clusters_map[pe].get_handle());
        }
        return handles;
    });

    parallel::vector_ets<std::size_t> size_ets(size);

    graph.pfor_nodes_range([&](const auto r) {
        auto& handles = used_clusters_map_handles.local();
        auto& size    = size_ets.local();

        for (NodeID u = r.begin(); u != r.end(); ++u) {
            const GlobalNodeID u_cluster                  = clustering[u];
            const auto [u_cluster_owner, u_local_cluster] = resolve_cluster_callback(u_cluster);

            auto [it, new_element] = handles[u_cluster_owner].insert(u_local_cluster + 1, 1);
            if (new_element) {
                handles[u_cluster_owner].update(u_local_cluster + 1, [&, u_cluster_owner = u_cluster_owner](auto& lhs) {
                    return lhs = 1 + next_slot_for_pe[u_cluster_owner]++;
                });
                ++size[u_cluster_owner];
            }
        }
    });

    std::vector<UsedClustersVector> used_clusters(size);

    auto sizes = size_ets.combine(std::plus{});
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
        used_clusters[pe].resize(sizes[pe]);
        auto handle = used_clusters_map[pe].get_handle();
        auto it     = handle.range(0, handle.capacity());
        while (it != handle.range_end()) {
            used_clusters[pe][(*it).second - 1] = (*it).first - 1;
            ++it;
        }
    });

    return {std::move(used_clusters_map), std::move(used_clusters)};
}

// global mapping, global number of coarse nodes
struct MappingResult {
    GlobalMapping                 mapping;
    scalable_vector<GlobalNodeID> distribution;
};

/*!
 * Compute a label mapping from fine nodes to coarse nodes.
 * @param graph The distributed graph.
 * @param clustering The global clustering to be contracted.
 * @return Label mapping and coarse node distribution. The coarse node distribution is such that coarse nodes are placed
 * on the PE which owned the corresponding cluster ID, i.e., if cluster ID \c x is owned by PE \c y, the coarse node ID
 * \c x is mapped to is also owned by PE \c y.
 */
MappingResult compute_mapping(
    const DistributedGraph& graph, const scalable_vector<parallel::Atomic<GlobalNodeID>>& clustering,
    const bool migrate_nodes = false
) {
    SCOPED_TIMER("Compute coarse node mapping");

    const auto size = mpi::get_comm_size(graph.communicator());
    const auto rank = mpi::get_comm_rank(graph.communicator());

    SCOPED_TIMER("first");

    START_TIMER("find_used_cluster_ids_per_pe");
    auto used_clusters = find_used_cluster_ids_per_pe(graph, clustering, [&](const GlobalNodeID cluster) {
        if (graph.is_owned_global_node(cluster)) {
            return std::make_pair(rank, graph.global_to_local_node(cluster));
        } else {
            const PEID owner = graph.find_owner_of_global_node(cluster);
            const auto local = static_cast<NodeID>(cluster - graph.offset_n(owner));
            return std::make_pair(owner, local);
        }
    });
    STOP_TIMER();

    SCOPED_TIMER("second");

    START_TIMER("bind references");
    auto& used_clusters_map = used_clusters.first;
    auto& used_clusters_vec = used_clusters.second;
    STOP_TIMER();

    SCOPED_TIMER("third");

    // send each PE its local node IDs that are used as cluster IDs somewhere
    START_TIMER("sparse_alltoall_get");
    const auto in_msg = mpi::sparse_alltoall_get<NodeID>(std::move(used_clusters_vec), graph.communicator());
    STOP_TIMER();

    SCOPED_TIMER("rest of the function");

    // map local labels to consecutive coarse node IDs
    scalable_vector<parallel::Atomic<GlobalNodeID>> label_mapping(graph.total_n());
    parallel::chunked_for(in_msg, [&](const NodeID local_label) {
        KASSERT(local_label < graph.n());
        label_mapping[local_label].store(1, std::memory_order_relaxed);
    });
    parallel::prefix_sum(label_mapping.begin(), label_mapping.end(), label_mapping.begin());

    const NodeID c_n = label_mapping.empty() ? 0 : static_cast<NodeID>(label_mapping.back());
    DBG << "Number of coarse nodes: " << c_n;

    // send mapping to other PEs that use cluster IDs from this PE -- i.e., answer in_msg
    std::vector<scalable_vector<NodeID>> out_msg(size);
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
        out_msg[pe].resize(in_msg[pe].size());
        tbb::parallel_for<std::size_t>(0, in_msg[pe].size(), [&](const std::size_t i) {
            KASSERT(in_msg[pe][i] < label_mapping.size());
            out_msg[pe][i] =
                label_mapping[in_msg[pe][i]] - 1; // label_mapping is 1-based due to the prefix sum operation
        });
    });

    const auto label_remap = mpi::sparse_alltoall_get<NodeID>(std::move(out_msg), graph.communicator());

    // migrate nodes from overloaded PEs
    scalable_vector<GlobalNodeID> c_distribution =
        create_distribution_from_local_count<GlobalNodeID>(c_n, graph.communicator());
    scalable_vector<GlobalNodeID> perfect_distribution{};
    scalable_vector<GlobalNodeID> pe_overload{};
    scalable_vector<GlobalNodeID> pe_underload{};

    SCOPED_TIMER("C");
    if (migrate_nodes) {
        const GlobalNodeID global_c_n = c_distribution.back();
        perfect_distribution          = create_perfect_distribution_from_global_count(global_c_n, graph.communicator());

        // compute diff between perfect distribution and current distribution
        pe_overload.resize(size + 1);
        pe_underload.resize(size + 1);

        scalable_vector<GlobalNodeID> pe_overload_tmp(size);
        scalable_vector<GlobalNodeID> pe_underload_tmp(size);

        for (PEID pe = 0; pe < size; ++pe) {
            const auto [from, to]     = math::compute_local_range<GlobalNodeID>(global_c_n, size, pe);
            const auto balanced_count = static_cast<NodeID>(to - from);
            const auto actual_count   = static_cast<NodeID>(c_distribution[pe + 1] - c_distribution[pe]);

            if (balanced_count > actual_count) {
                pe_underload_tmp[pe] = balanced_count - actual_count;
            } else {
                pe_overload_tmp[pe] = actual_count - balanced_count;
            }
        }

        // prefix sums allow us to find the new owner of a migrating node in log time using binary search
        parallel::prefix_sum(pe_overload_tmp.begin(), pe_overload_tmp.end(), pe_overload.begin() + 1);
        parallel::prefix_sum(pe_underload_tmp.begin(), pe_underload_tmp.end(), pe_underload.begin() + 1);
    }

    tbb::enumerable_thread_specific<std::vector<UsedClustersMap::handle_type>> used_clusters_ets([&] {
        std::vector<UsedClustersMap::handle_type> handles;
        handles.reserve(size);
        for (PEID pe = 0; pe < size; ++pe) {
            handles.push_back(used_clusters_map[pe].get_handle());
        }
        return handles;
    });

    // now  we use label_mapping as a [fine node -> coarse node] mapping of local nodes on this PE -- and extend it
    // for ghost nodes in the next step
    // all cluster[.] labels are stored in label_remap, thus we can overwrite label_mapping
    graph.pfor_nodes([&](const NodeID u) {
        const GlobalNodeID u_cluster = clustering[u];
        PEID               u_cluster_owner;
        NodeID             u_local_cluster;

        if (graph.is_owned_global_node(u_cluster)) {
            u_cluster_owner = rank;
            u_local_cluster = graph.global_to_local_node(u_cluster);
        } else {
            u_cluster_owner = graph.find_owner_of_global_node(u_cluster);
            u_local_cluster = static_cast<NodeID>(u_cluster - graph.offset_n(u_cluster_owner));
        }

        auto& handles = used_clusters_ets.local();
        auto  it      = handles[u_cluster_owner].find(u_local_cluster + 1);
        KASSERT(it != handles[u_cluster_owner].end());

        const NodeID slot_in_msg = (*it).second - 1;
        const NodeID label       = label_remap[u_cluster_owner][slot_in_msg];

        if (migrate_nodes) {
            const auto count =
                static_cast<NodeID>(perfect_distribution[u_cluster_owner + 1] - perfect_distribution[u_cluster_owner]);
            if (label < count) { // node can stay on PE
                label_mapping[u] = perfect_distribution[u_cluster_owner] + label;
            } else { // move node to another PE
                const GlobalNodeID position = pe_overload[u_cluster_owner] + label - count;
                const PEID         new_owner =
                    static_cast<PEID>(math::find_in_distribution<GlobalNodeID>(position, pe_underload));

                KASSERT(position >= pe_underload[new_owner]);
                KASSERT(
                    perfect_distribution[new_owner + 1] - perfect_distribution[new_owner]
                    > c_distribution[new_owner + 1] - c_distribution[new_owner]
                );

                label_mapping[u] = perfect_distribution[new_owner] + c_distribution[new_owner + 1]
                                   - c_distribution[new_owner] + position - pe_underload[new_owner];
            }
        } else {
            label_mapping[u] = c_distribution[u_cluster_owner] + label;
        }
    });

    // exchange labels for ghost nodes
    exchange_ghost_node_mapping(graph, label_mapping);

    if (migrate_nodes) {
        c_distribution = std::move(perfect_distribution);
    }

    SCOPED_TIMER("end");
    return {std::move(label_mapping), std::move(c_distribution)};
}

/*!
 * Construct the coarse graph.
 * @tparam CoarseNodeOwnerCallback
 * @param graph The distributed graph to be contracted.
 * @param mapping Label mapping from fine to coarse nodes.
 * @param c_node_distribution Coarse node distribution: determines which coarse nodes are owned by which PEs using the
 * lambda callback.
 * @param compute_coarse_node_owner Determines which coarse node is owned by which PE: this could be computed using
 * binary search on \c c_node_distribution, but based on the coarse node distribution, the PE could also be computed
 * in constant time.
 * @return The distributed coarse graph.
 */
template <typename CoarseNodeOwnerCallback, typename Mapping>
DistributedGraph build_coarse_graph(
    const DistributedGraph& graph, const Mapping& mapping, scalable_vector<GlobalNodeID> c_node_distribution,
    CoarseNodeOwnerCallback&& compute_coarse_node_owner
) {
    SCOPED_TIMER("Build coarse graph");

    const PEID size = mpi::get_comm_size(graph.communicator());
    const PEID rank = mpi::get_comm_rank(graph.communicator());

    // compute coarse node distribution
    const auto from = c_node_distribution[rank];
    const auto to   = c_node_distribution[rank + 1];

    // create messages
    std::vector<NoinitVector<LocalToGlobalEdge>> out_msg(size); // declare outside scope
    {
        SCOPED_TIMER("Create edge messages");
        const PEID                                num_threads = omp_get_max_threads();
        std::vector<cache_aligned_vector<EdgeID>> num_messages(num_threads, cache_aligned_vector<EdgeID>(size));

        START_TIMER("Count messages");
#pragma omp parallel for default(none) \
    shared(num_messages, graph, mapping, compute_coarse_node_owner, c_node_distribution)
        for (NodeID u = 0; u < graph.n(); ++u) {
            const PEID thread    = omp_get_thread_num();
            const auto c_u       = mapping[u];
            const auto c_u_owner = compute_coarse_node_owner(c_u, c_node_distribution);

            // for (EdgeID e = graph.first_edge(u); e < graph.first_invalid_edge(u); ++e) {
            for (const auto [e, v]: graph.neighbors(u)) {
                // const auto v = graph.edge_target(e);
                const auto c_v        = mapping[v];
                const bool is_message = c_u != c_v;
                num_messages[thread][c_u_owner] += is_message;
            }
        }

        mpi::graph::internal::inclusive_col_prefix_sum(num_messages); // TODO move this utility function somewhere else
        STOP_TIMER();

        // allocate send buffers
        START_TIMER("Allocation");
        tbb::parallel_for<PEID>(0, size, [&](const PEID pe) { out_msg[pe].resize(num_messages.back()[pe]); });
        STOP_TIMER();

        START_TIMER("Create messages");
#pragma omp parallel for default(none) \
    shared(num_messages, graph, mapping, compute_coarse_node_owner, c_node_distribution, out_msg)
        for (NodeID u = 0; u < graph.n(); ++u) {
            const PEID thread    = omp_get_thread_num();
            const auto c_u       = mapping[u];
            const auto c_u_owner = compute_coarse_node_owner(c_u, c_node_distribution);
            const auto local_c_u = static_cast<NodeID>(c_u - c_node_distribution[c_u_owner]);

            for (const auto [e, v]: graph.neighbors(u)) {
                const auto c_v = mapping[v];

                if (c_u != c_v) { // ignore self loops
                    const std::size_t slot   = --num_messages[thread][c_u_owner];
                    out_msg[c_u_owner][slot] = {.u = local_c_u, .weight = graph.edge_weight(e), .v = c_v};
                }
            }
        }
        STOP_TIMER();
    }

    // deduplicate edges
    TIMED_SCOPE("Deduplicate edges before sending") {
        DeduplicateEdgeListMemoryContext deduplicate_m_ctx;
        for (PEID pe = 0; pe < size; ++pe) {
            auto result       = deduplicate_edge_list_parallel(std::move(out_msg[pe]), std::move(deduplicate_m_ctx));
            out_msg[pe]       = std::move(result.first);
            deduplicate_m_ctx = std::move(result.second);
        }
    };

    // exchange messages
    START_TIMER("Exchange edges");
    auto in_msg = mpi::sparse_alltoall_get<LocalToGlobalEdge>(std::move(out_msg), graph.communicator());
    STOP_TIMER();

    // Copy edge lists to a single list and free old list
    START_TIMER("Copy edge list");
    std::vector<std::size_t> in_msg_sizes(size);
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) { in_msg_sizes[pe] = in_msg[pe].size(); });
    parallel::prefix_sum(in_msg_sizes.begin(), in_msg_sizes.end(), in_msg_sizes.begin());

    START_TIMER("Allocation");
    NoinitVector<LocalToGlobalEdge> edge_list(in_msg_sizes.back());
    STOP_TIMER();

    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
        tbb::parallel_for<std::size_t>(0, in_msg[pe].size(), [&](const std::size_t i) {
            edge_list[in_msg_sizes[pe] - in_msg[pe].size() + i] = in_msg[pe][i];
        });
        // std::copy(in_msg[pe].begin(), in_msg[pe].end(), edge_list.begin() + in_msg_sizes[pe] - in_msg[pe].size());
    });
    STOP_TIMER();

    // TODO since we do not know the number of coarse ghost nodes yet, allocate memory only for local nodes and
    // TODO resize in build_distributed_graph_from_edge_list
    KASSERT(from <= to);
    scalable_vector<parallel::Atomic<NodeWeight>> node_weights(to - from);
    struct NodeWeightMessage {
        NodeID     node;
        NodeWeight weight;
    };

    START_TIMER("Exchange node weights");
    mpi::graph::sparse_alltoall_custom<NodeWeightMessage>(
        graph, 0, graph.n(), SPARSE_ALLTOALL_NOFILTER,
        [&](const NodeID u) { return compute_coarse_node_owner(mapping[u], c_node_distribution); },
        [&](const NodeID u) -> NodeWeightMessage {
            const auto   c_u       = mapping[u];
            const PEID   c_u_owner = compute_coarse_node_owner(c_u, c_node_distribution);
            const NodeID c_u_local = c_u - c_node_distribution[c_u_owner];
            return {c_u_local, graph.node_weight(u)};
        },
        [&](const auto r) {
            tbb::parallel_for<std::size_t>(0, r.size(), [&](const std::size_t i) {
                node_weights[r[i].node].fetch_add(r[i].weight, std::memory_order_relaxed);
            });
        }
    );
    STOP_TIMER();

    // now every PE has an edge list with all edges -- so we can build the graph from it
    return build_distributed_graph_from_edge_list(
        edge_list, std::move(c_node_distribution), graph.communicator(),
        [&](const NodeID u) {
            KASSERT(u < node_weights.size());
            return node_weights[u].load(std::memory_order_relaxed);
        },
        compute_coarse_node_owner
    );
}

/*!
 * Construct the coarse graph.
 * @param graph The distributed graph to be contracted.
 * @param mapping Label mapping from fine to coarse nodes.
 * @param c_node_distribution Coarse node distribution: determines which coarse nodes are owned by which PEs using
 * binary search.
 * @return The distributed coarse graph.
 */
template <typename Mapping>
DistributedGraph build_coarse_graph(
    const DistributedGraph& graph, const Mapping& mapping, scalable_vector<GlobalNodeID> c_node_distribution
) {
    return build_coarse_graph(
        graph, mapping, c_node_distribution,
        [](const GlobalNodeID node, const auto& node_distribution) {
            const auto it = std::upper_bound(node_distribution.begin() + 1, node_distribution.end(), node);
            return static_cast<PEID>(std::distance(node_distribution.begin(), it) - 1);
        }
    );
}

/*!
 * Sparse all-to-all to update ghost node weights after coarse graph construction.
 * @param graph Distributed graph with invalid ghost node weights.
 */
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
} // namespace

//! Contract a distributed graph such that coarse nodes are owned by the PE which owned the respective cluster ID.
GlobalContractionResult
contract_global_clustering_no_migration(const DistributedGraph& graph, const GlobalClustering& clustering) {
    SCOPED_TIMER("Contract clustering");

    auto [mapping, distribution] = compute_mapping(graph, clustering);
    auto c_graph                 = build_coarse_graph(graph, mapping, std::move(distribution));
    update_ghost_node_weights(c_graph);

    return {std::move(c_graph), std::move(mapping)};
}

//! Contract a distributed graph such that *most* coarse nodes are owned by the PE which owned the respective cluster
//! ID, while migrating enough coarse nodes such that each PE ownes approx. the same number of coarse nodes.
GlobalContractionResult
contract_global_clustering_minimal_migration(const DistributedGraph& graph, const GlobalClustering& clustering) {
    SCOPED_TIMER("Contract clustering");

    auto [mapping, distribution] = compute_mapping(graph, clustering, true);
    auto c_graph                 = build_coarse_graph(graph, mapping, std::move(distribution));
    update_ghost_node_weights(c_graph);

    return {std::move(c_graph), std::move(mapping)};
}

//! Contract a distributed graph such that each PE owns the same number of coarse nodes by assigning coarse nodes
//! \code{p*n/s .. (p + 1)*n/s} to PE \c p, where \c n is the number of coarse nodes and \c s is the number of PEs.
GlobalContractionResult
contract_global_clustering_full_migration(const DistributedGraph& graph, const GlobalClustering& clustering) {
    SCOPED_TIMER("Contract clustering");

    auto [mapping, distribution] = compute_mapping(graph, clustering);

    // create a new node distribution where nodes are evenly distributed across PEs
    const PEID         size       = mpi::get_comm_size(graph.communicator());
    const GlobalNodeID c_global_n = distribution.back();
    auto               c_graph    = build_coarse_graph(
        graph, mapping, create_perfect_distribution_from_global_count<GlobalNodeID>(c_global_n, graph.communicator()),
        [size, c_global_n](const GlobalNodeID node, const auto& /* node_distribution */) {
            return math::compute_local_range_rank<GlobalNodeID>(c_global_n, size, node);
        }
    );

    update_ghost_node_weights(c_graph);

    return {std::move(c_graph), std::move(mapping)};
}

GlobalContractionResult contract_global_clustering(
    const DistributedGraph& graph, const GlobalClustering& clustering, const GlobalContractionAlgorithm algorithm
) {
    SCOPED_TIMER("Contract clustering");

    switch (algorithm) {
        case GlobalContractionAlgorithm::NO_MIGRATION:
            return contract_global_clustering_no_migration(graph, clustering);
        case GlobalContractionAlgorithm::MINIMAL_MIGRATION:
            return contract_global_clustering_minimal_migration(graph, clustering);
        case GlobalContractionAlgorithm::FULL_MIGRATION:
            return contract_global_clustering_full_migration(graph, clustering);
        case GlobalContractionAlgorithm::V2: {
            auto [c_graph, c_mapping] = contract_clustering(graph, clustering);
            GlobalMapping c_mapping2(c_mapping.size());
            std::copy(c_mapping.begin(), c_mapping.end(), c_mapping2.begin());
            return {std::move(c_graph), std::move(c_mapping2)};
        }
    }
    __builtin_unreachable();
}

/*!
 * Projects the partition of the coarse graph onto the fine graph. Works for any graph contraction variations.
 * @param fine_graph The distributed fine graph.
 * @param coarse_graph The distributed coarse graph with partition.
 * @param fine_to_coarse Mapping from fine to coarse nodes.
 * @return Projected partition of the fine graph.
 */
DistributedPartitionedGraph project_global_contracted_graph(
    const DistributedGraph& fine_graph, DistributedPartitionedGraph coarse_graph, const GlobalMapping& fine_to_coarse
) {
    SCOPED_TIMER("Project partition");

    const PEID size = mpi::get_comm_size(fine_graph.communicator());

    // find unique coarse_graph node IDs of fine_graph nodes
    auto resolve_coarse_node = [&](const GlobalNodeID coarse_node) {
        KASSERT(coarse_node < coarse_graph.global_n());
        const PEID owner = coarse_graph.find_owner_of_global_node(coarse_node);
        const auto local = static_cast<NodeID>(coarse_node - coarse_graph.offset_n(owner));
        return std::make_pair(owner, local);
    };

    auto used_coarse_nodes = find_used_cluster_ids_per_pe(fine_graph, fine_to_coarse, resolve_coarse_node);

    auto& used_coarse_nodes_map = used_coarse_nodes.first;
    auto& used_coarse_nodes_vec = used_coarse_nodes.second;

    // send requests for block IDs
    const auto reqs = mpi::sparse_alltoall_get<NodeID>(used_coarse_nodes_vec, fine_graph.communicator());

    // build response messages
    START_TIMER("Allocation");
    std::vector<scalable_vector<BlockID>> resps(size);
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) { resps[pe].resize(reqs[pe].size()); });
    STOP_TIMER();

    START_TIMER("Build response messages");
    tbb::parallel_for<std::size_t>(0, reqs.size(), [&](const std::size_t i) {
        tbb::parallel_for<std::size_t>(0, reqs[i].size(), [&](const std::size_t j) {
            KASSERT(coarse_graph.is_owned_node(reqs[i][j]));
            resps[i][j] = coarse_graph.block(reqs[i][j]);
        });
    });
    STOP_TIMER();

    // exchange messages and use used_coarse_nodes_map to store block IDs
    tbb::enumerable_thread_specific<std::vector<UsedClustersMap::handle_type>> used_coarse_nodes_ets([&] {
        std::vector<UsedClustersMap::handle_type> handles;
        handles.reserve(size);
        for (PEID pe = 0; pe < size; ++pe) {
            handles.push_back(used_coarse_nodes_map[pe].get_handle());
        }
        return handles;
    });

    static_assert(std::numeric_limits<BlockID>::digits <= std::numeric_limits<NodeID>::digits);
    mpi::sparse_alltoall<BlockID>(
        std::move(resps),
        [&](const auto buffer, const PEID pe) {
            tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
                auto& handles = used_coarse_nodes_ets.local();

                KASSERT(static_cast<std::size_t>(pe) < used_coarse_nodes_map.size());
                KASSERT(static_cast<std::size_t>(pe) < reqs.size());
                KASSERT(i < used_coarse_nodes_vec[pe].size());

                const auto [it, found] = handles[pe].update(
                    used_coarse_nodes_vec[pe][i] + 1, [&](auto& lhs, const auto rhs) { return lhs = rhs; },
                    buffer[i] + 1
                );
                KASSERT(found);
            });
        },
        fine_graph.communicator()
    );

    // assign block IDs to fine nodes
    START_TIMER("Allocation");
    scalable_vector<BlockID> fine_partition(fine_graph.total_n());
    STOP_TIMER();

    START_TIMER("Set blocks");
    fine_graph.pfor_nodes([&](const NodeID u) {
        const auto [owner, local] = resolve_coarse_node(fine_to_coarse[u]);

        auto& handles = used_coarse_nodes_ets.local();

        auto it = handles[owner].find(local + 1);
        KASSERT(it != handles[owner].end());

        fine_partition[u] = (*it).second - 1;
    });
    STOP_TIMER();

    // exchange ghost node labels
    struct GhostNodeLabel {
        NodeID  local_node_on_sender;
        BlockID block;
    };

    mpi::graph::sparse_alltoall_interface_to_pe<GhostNodeLabel>(
        fine_graph,
        [&](const NodeID u) -> GhostNodeLabel {
            return {u, fine_partition[u]};
        },
        [&](const auto buffer, const PEID pe) {
            tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
                const auto& [local_node_on_sender, block] = buffer[i];
                const GlobalNodeID global_node            = fine_graph.offset_n(pe) + local_node_on_sender;
                const NodeID       local_node             = fine_graph.global_to_local_node(global_node);
                fine_partition[local_node]                = block;
            });
        }
    );

    return {&fine_graph, coarse_graph.k(), std::move(fine_partition), coarse_graph.take_block_weights()};
}

ContractionResult contract_clustering(const DistributedGraph& graph, const GlobalClustering& clustering) {
    SET_DEBUG(false);

    const PEID size = mpi::get_comm_size(graph.communicator());
    const PEID rank = mpi::get_comm_rank(graph.communicator());

    //
    // Collect nodes and edges that must be migrated to another PE
    //
    START_TIMER("Collect nodes and edges for other PEs");
    NoinitVector<NodeID> edge_position_buffer(graph.n() + 1);
    NoinitVector<NodeID> node_position_buffer(graph.n() + 1);
    edge_position_buffer.front() = 0;
    node_position_buffer.front() = 0;

    graph.pfor_nodes([&](const NodeID u) {
        const GlobalNodeID c_u = clustering[u];

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

    struct Edge {
        GlobalNodeID u;
        GlobalNodeID v;
        EdgeWeight   weight;
    };

    struct Node {
        GlobalNodeID u;
        NodeWeight   weight;
    };

    NoinitVector<Edge> nonlocal_edges(edge_position_buffer.back());
    NoinitVector<Node> nonlocal_nodes(node_position_buffer.back());

    graph.pfor_nodes([&](const NodeID u) {
        const GlobalNodeID c_u = clustering[u];

        if (!graph.is_owned_global_node(c_u)) {
            // Node
            nonlocal_nodes[node_position_buffer[u]] = {.u = c_u, .weight = graph.node_weight(u)};

            // Edge
            std::size_t pos = edge_position_buffer[u];
            for (const auto [e, v]: graph.neighbors(u)) {
                nonlocal_edges[pos] = {
                    .u      = c_u,
                    .v      = clustering[v],
                    .weight = graph.edge_weight(e),
                };
                ++pos;
            }
        }
    });
    STOP_TIMER();

    //
    // Deduplicate the edges before sending them
    //
    START_TIMER("Deduplicate edges before sending");
    if (!nonlocal_edges.empty()) {
        // Primary sort by edge source = messages are sorted by destination PE
        // Secondary sort by edge target = duplicate edges are consecutive
        tbb::parallel_sort(nonlocal_edges.begin(), nonlocal_edges.end(), [&](const auto& lhs, const auto& rhs) {
            return lhs.u < rhs.u || (lhs.u == rhs.u && lhs.v < rhs.v);
        });

        // Mark the first edge in every block of duplicate edges
        NoinitVector<EdgeID> edge_position_buffer(nonlocal_edges.size());
        tbb::parallel_for<std::size_t>(0, nonlocal_edges.size(), [&](const std::size_t i) {
            edge_position_buffer[i] = 0;
        });
        tbb::parallel_for<std::size_t>(1, nonlocal_edges.size(), [&](const std::size_t i) {
            if (nonlocal_edges[i].u != nonlocal_edges[i - 1].u || nonlocal_edges[i].v != nonlocal_edges[i - 1].v) {
                edge_position_buffer[i] = 1;
            }
        });

        // Prefix sum to get the location of the deduplicated edge
        parallel::prefix_sum(edge_position_buffer.begin(), edge_position_buffer.end(), edge_position_buffer.begin());

        // Deduplicate edges in a separate buffer
        NoinitVector<Edge> tmp_nonlocal_edges(edge_position_buffer.back() + 1);
        tbb::parallel_for<std::size_t>(0, edge_position_buffer.back() + 1, [&](const std::size_t i) {
            tmp_nonlocal_edges[i].weight = 0;
        });
        tbb::parallel_for<std::size_t>(0, nonlocal_edges.size(), [&](const std::size_t i) {
            const std::size_t pos = edge_position_buffer[i];
            __atomic_store_n(&(tmp_nonlocal_edges[pos].u), nonlocal_edges[i].u, __ATOMIC_RELAXED);
            __atomic_store_n(&(tmp_nonlocal_edges[pos].v), nonlocal_edges[i].v, __ATOMIC_RELAXED);
            __atomic_fetch_add(&(tmp_nonlocal_edges[pos].weight), nonlocal_edges[i].weight, __ATOMIC_RELAXED);
        });
        std::swap(tmp_nonlocal_edges, nonlocal_edges);
    }

    // Also sort the nodes so that we can count them efficiently in the next step
    tbb::parallel_sort(nonlocal_nodes.begin(), nonlocal_nodes.end(), [&](const auto& lhs, const auto& rhs) {
        return lhs.u < rhs.u;
    });
    STOP_TIMER();

    //
    // Exchange nodes and edges
    //

    // First, count them
    START_TIMER("Migrate nodes and edges");
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
    NoinitVector<Edge> local_edges;
    std::vector<int>   local_edges_sendcounts(size);
    std::vector<int>   local_edges_sdispls(size);
    std::vector<int>   local_edges_recvcounts(size);
    std::vector<int>   local_edges_rdispls(size);

    std::copy(num_edges_for_pe.begin(), num_edges_for_pe.end(), local_edges_sendcounts.begin());
    std::exclusive_scan(local_edges_sendcounts.begin(), local_edges_sendcounts.end(), local_edges_sdispls.begin(), 0);
    MPI_Alltoall(
        local_edges_sendcounts.data(), 1, MPI_INT, local_edges_recvcounts.data(), 1, MPI_INT, graph.communicator()
    );
    std::exclusive_scan(local_edges_recvcounts.begin(), local_edges_recvcounts.end(), local_edges_rdispls.begin(), 0);

    local_edges.resize(local_edges_rdispls.back() + local_edges_recvcounts.back());
    MPI_Alltoallv(
        nonlocal_edges.data(), local_edges_sendcounts.data(), local_edges_sdispls.data(), mpi::type::get<Edge>(),
        local_edges.data(), local_edges_recvcounts.data(), local_edges_rdispls.data(), mpi::type::get<Edge>(),
        graph.communicator()
    );

    // Sort edges
    tbb::parallel_sort(local_edges.begin(), local_edges.end(), [&](const auto& lhs, const auto& rhs) {
        return lhs.u < rhs.u;
    });

    if constexpr (kDebug) {
        std::stringstream ss;
        for (const auto& edge: local_edges) {
            ss << " {" << edge.u << ", " << edge.v << ", " << edge.weight << "}";
        }
        DBG << "Received edges: " << ss.str();
    }

    // Exchange nodes

    NoinitVector<Node> local_nodes;
    std::vector<int>   local_nodes_sendcounts(size);
    std::vector<int>   local_nodes_sdispls(size);
    std::vector<int>   local_nodes_recvcounts(size);
    std::vector<int>   local_nodes_rdispls(size);

    std::copy(num_nodes_for_pe.begin(), num_nodes_for_pe.end(), local_nodes_sendcounts.begin());
    std::exclusive_scan(local_nodes_sendcounts.begin(), local_nodes_sendcounts.end(), local_nodes_sdispls.begin(), 0);
    MPI_Alltoall(
        local_nodes_sendcounts.data(), 1, MPI_INT, local_nodes_recvcounts.data(), 1, MPI_INT, graph.communicator()
    );
    std::exclusive_scan(local_nodes_recvcounts.begin(), local_nodes_recvcounts.end(), local_nodes_rdispls.begin(), 0);

    if constexpr (kDebug) {
        std::stringstream ss;
        for (const auto& node: nonlocal_nodes) {
            ss << " {" << node.u << ", " << node.weight << "}";
        }
        DBG << "Sending nodes: " << ss.str() << " ::: sendcounts=[" << local_nodes_sendcounts << "], recvcounts=["
            << local_nodes_recvcounts << "]";
    }

    local_nodes.resize(local_nodes_rdispls.back() + local_nodes_recvcounts.back());
    MPI_Alltoallv(
        nonlocal_nodes.data(), local_nodes_sendcounts.data(), local_nodes_sdispls.data(), mpi::type::get<Node>(),
        local_nodes.data(), local_nodes_recvcounts.data(), local_nodes_rdispls.data(), mpi::type::get<Node>(),
        graph.communicator()
    );

    if constexpr (kDebug) {
        std::stringstream ss;
        for (const auto& node: local_nodes) {
            ss << " {" << node.u << ", " << node.weight << "}";
        }
        DBG << "Received nodes: " << ss.str();
    }
    STOP_TIMER();

    //
    // Next, build the coarse node distribution
    //

    // Map non-empty clusters belonging to this PE to a consecutive range of coarse node IDs:
    // ```
    // clustering_mapping[local node ID] = local coarse node ID
    // ```
    START_TIMER("Build mapping");
    NoinitVector<NodeID> cluster_mapping(graph.n());
    graph.pfor_nodes([&](const NodeID u) { cluster_mapping[u] = 0; });
    tbb::parallel_invoke(
        [&] {
            graph.pfor_nodes([&](const NodeID u) {
                const GlobalNodeID c_u = clustering[u];
                if (graph.is_owned_global_node(c_u)) {
                    const NodeID local_cluster = static_cast<NodeID>(c_u - graph.offset_n());
                    __atomic_store_n(&cluster_mapping[local_cluster], 1, __ATOMIC_RELAXED);
                }
            });
        },
        [&] {
            tbb::parallel_for<std::size_t>(0, local_nodes.size(), [&](const std::size_t i) {
                const GlobalNodeID c_u = local_nodes[i].u;
                KASSERT(graph.is_owned_global_node(c_u), V(c_u));
                const NodeID local_cluster = static_cast<NodeID>(c_u - graph.offset_n());
                __atomic_store_n(&cluster_mapping[local_cluster], 1, __ATOMIC_RELAXED);
            });
        }
    );
    parallel::prefix_sum(cluster_mapping.begin(), cluster_mapping.end(), cluster_mapping.begin());
    STOP_TIMER();

    // Number of coarse nodes on this PE:
    const NodeID c_n = cluster_mapping.empty() ? 0 : cluster_mapping.back();
    DBG << "Number of coarse nodes: " << c_n;

    // Make cluster IDs start at 0
    START_TIMER("Build coarse node distribution");
    tbb::parallel_for<std::size_t>(0, cluster_mapping.size(), [&](const std::size_t i) { cluster_mapping[i] -= 1; });

    scalable_vector<GlobalNodeID> c_node_distribution(size + 1);
    MPI_Allgather(
        &c_n, 1, mpi::type::get<NodeID>(), c_node_distribution.data(), 1, mpi::type::get<GlobalNodeID>(),
        graph.communicator()
    );
    std::exclusive_scan(c_node_distribution.begin(), c_node_distribution.end(), c_node_distribution.begin(), 0u);
    STOP_TIMER();
    DBG << "Coarse node distribution: [" << c_node_distribution << "]";

    START_TIMER("Exchange mapping of migrated nodes");
    struct NodeMapping {
        GlobalNodeID u;
        GlobalNodeID global_c_u;
    };
    NoinitVector<NodeMapping> local_nodes_mapping(local_nodes.size());
    tbb::parallel_for<std::size_t>(0, local_nodes.size(), [&](const std::size_t i) {
        local_nodes_mapping[i] = {
            .u          = local_nodes[i].u,
            .global_c_u = cluster_mapping[local_nodes[i].u - graph.offset_n()] + c_node_distribution[rank],
        };
    });
    NoinitVector<NodeMapping> local_nodes_mapping_rsps(nonlocal_nodes.size());
    MPI_Alltoallv(
        local_nodes_mapping.data(), local_nodes_recvcounts.data(), local_nodes_rdispls.data(),
        mpi::type::get<NodeMapping>(), local_nodes_mapping_rsps.data(), local_nodes_sendcounts.data(),
        local_nodes_sdispls.data(), mpi::type::get<NodeMapping>(), graph.communicator()
    );
    STOP_TIMER();

    // We can now map fine nodes to coarse nodes if their respective cluster is owned by this PE
    // For other nodes, we have to communicate to determine their coarse node ID
    // There are two types of nodes that we need this mapping for:
    // - Coarse ghost nodes
    // X Local nodes that were migrated to another PE (to project the coarse partition onto the finer graph later)

    START_TIMER("Communicate mapping for ghost nodes");
    // Build a list of global nodes for which we need their new coarse node ID
    using NonlocalClusterFilter = growt::GlobalNodeIDMap<GlobalNodeID>;
    NonlocalClusterFilter nonlocal_cluster_filter(0); // graph.n() == 0 ? 0 : 1.0 * graph.ghost_n() / graph.n() * c_n);
    tbb::enumerable_thread_specific<NonlocalClusterFilter::handle_type> nonlocal_cluster_filter_handle_ets([&] {
        return nonlocal_cluster_filter.get_handle();
    });

    std::vector<parallel::Atomic<NodeID>> next_index_for_pe(size + 1);

    auto request_nonlocal_mapping = [&](const GlobalNodeID cluster) {
        auto& handle          = nonlocal_cluster_filter_handle_ets.local();
        const auto [it, mine] = handle.insert(cluster + 1, 1); // dummy value
        if (mine) {
            const PEID owner = graph.find_owner_of_global_node(cluster);
            handle.update(cluster + 1, [&](auto& lhs) { return lhs = ++next_index_for_pe[owner]; });
        }
    };

    tbb::parallel_invoke(
        [&] {
            graph.pfor_nodes([&](const NodeID u) {
                const GlobalNodeID cluster_u = clustering[u];
                if (!graph.is_owned_global_node(cluster_u)) {
                    // request_nonlocal_mapping(cluster_u);
                    return;
                }

                for (const auto [e, v]: graph.neighbors(u)) {
                    const GlobalNodeID cluster_v = clustering[v];
                    if (!graph.is_owned_global_node(cluster_v)) {
                        request_nonlocal_mapping(cluster_v);
                    }
                }
            });
        },
        [&] {
            tbb::parallel_for<std::size_t>(0, local_edges.size(), [&](const std::size_t i) {
                const GlobalNodeID cluster_v = local_edges[i].v;
                if (!graph.is_owned_global_node(cluster_v)) {
                    request_nonlocal_mapping(cluster_v);
                }
            });
        }
    );

    std::vector<scalable_vector<GlobalNodeID>> my_mapping_requests(size);
    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) { my_mapping_requests[pe].resize(next_index_for_pe[pe]); });
    static std::atomic_size_t counter;
    counter = 0;

#pragma omp parallel
    {
        auto& handle = nonlocal_cluster_filter_handle_ets.local();

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
            const NodeID       coarse_local  = cluster_mapping[local];
            const GlobalNodeID coarse_global = c_node_distribution[rank] + coarse_local;
            my_mapping_responses[pe][i]      = coarse_global;
        });
    });

    auto their_mapping_responses = mpi::sparse_alltoall_get<GlobalNodeID>(my_mapping_responses, graph.communicator());
    STOP_TIMER();

    // Now we can build the coarse ghost node mapping
    START_TIMER("Build mapping");
    std::exclusive_scan(next_index_for_pe.begin(), next_index_for_pe.end(), next_index_for_pe.begin(), 0);
    const NodeID c_ghost_n = next_index_for_pe.back();

    growt::StaticGhostNodeMapping c_global_to_ghost(c_ghost_n);
    scalable_vector<GlobalNodeID> c_ghost_to_global(c_ghost_n);
    scalable_vector<PEID>         c_ghost_owner(c_ghost_n);

    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
        for (std::size_t i = 0; i < my_mapping_requests[pe].size(); ++i) {
            const GlobalNodeID global = their_mapping_responses[pe][i];
            const NodeID       local  = next_index_for_pe[pe] + i;
            c_global_to_ghost.insert(global + 1, c_n + local);
            c_ghost_to_global[local] = global;
            c_ghost_owner[local]     = pe;
        }
    });

    tbb::parallel_for(tbb::blocked_range<std::size_t>(0, local_nodes_mapping_rsps.size()), [&](const auto& r) {
        auto& handle = nonlocal_cluster_filter_handle_ets.local();
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
            const auto& [local_cluster, global_coarse_node] = local_nodes_mapping_rsps[i];
            DBG << "Insert " << local_cluster << " --> " << global_coarse_node;
            handle.insert(local_cluster + 1, graph.global_n() + global_coarse_node + 1);
        }
    });

    // Build a mapping array from fine nodes to coarse nodes
    NoinitVector<GlobalNodeID> mapping(graph.total_n());
    graph.pfor_all_nodes([&](const NodeID u) {
        const GlobalNodeID cluster = clustering[u];

        if (graph.is_owned_global_node(cluster)) {
            mapping[u] = cluster_mapping[cluster - graph.offset_n()] + c_node_distribution[rank];
            DBG << "a " << u << " " << V(mapping[u]);
        } else {
            auto& handle = nonlocal_cluster_filter_handle_ets.local();
            auto  it     = handle.find(cluster + 1);
            if (it != handle.end()) {
                const std::size_t index = (*it).second - 1;
                if (index < graph.global_n()) {
                    const PEID owner = graph.find_owner_of_global_node(cluster);
                    mapping[u]       = their_mapping_responses[owner][index];
                    DBG << "b " << u << " " << V(mapping[u]);
                } else {
                    mapping[u] = static_cast<GlobalNodeID>(index - graph.global_n());
                    // @todo local vs global mapping, create global mapping with this info
                    DBG << "c " << u << " " << V(mapping[u]);
                }
            }
        }

        KASSERT(mapping[u] < c_node_distribution.back());
    });
    STOP_TIMER();

    //
    // Sort local nodes by their cluster ID
    //
    START_TIMER("Bucket sort nodes by cluster");
    NoinitVector<NodeID> buckets_position_buffer(c_n + 1);
    tbb::parallel_for<NodeID>(0, c_n + 1, [&](const NodeID c_u) { buckets_position_buffer[c_u] = 0; });

    tbb::parallel_invoke(
        [&] {
            graph.pfor_nodes([&](const NodeID u) {
                const GlobalNodeID cluster = clustering[u];
                if (graph.is_owned_global_node(cluster)) {
                    const NodeID local_cluster = static_cast<NodeID>(cluster - graph.offset_n());
                    const NodeID c_u           = cluster_mapping[local_cluster];
                    KASSERT(c_u < buckets_position_buffer.size());
                    __atomic_fetch_add(&buckets_position_buffer[c_u], 1, __ATOMIC_RELAXED);
                }
            });
        },
        [&] {
            tbb::parallel_for<std::size_t>(0, local_edges.size(), [&](const std::size_t i) {
                const NodeID local_cluster = static_cast<NodeID>(local_edges[i].u - graph.offset_n());
                const NodeID c_u           = cluster_mapping[local_cluster];
                local_edges[i].u           = c_u;

                if (i == 0 || local_edges[i].u != local_edges[i - 1].u) {
                    __atomic_fetch_add(&buckets_position_buffer[c_u], 1, __ATOMIC_RELAXED);
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
            graph.pfor_nodes([&](const NodeID u) {
                const GlobalNodeID cluster = clustering[u];
                if (graph.is_owned_global_node(cluster)) {
                    const NodeID      local_cluster = static_cast<NodeID>(cluster - graph.offset_n());
                    const NodeID      c_u           = cluster_mapping[local_cluster];
                    const std::size_t pos = __atomic_fetch_sub(&buckets_position_buffer[c_u], 1, __ATOMIC_RELAXED);
                    buckets[pos - 1]      = u;
                }
            });
        },
        [&] {
            tbb::parallel_for<std::size_t>(0, local_edges.size(), [&](const std::size_t i) {
                if (i == 0 || local_edges[i].u != local_edges[i - 1].u) {
                    const NodeID      c_u = local_edges[i].u;
                    const std::size_t pos = __atomic_fetch_sub(&buckets_position_buffer[c_u], 1, __ATOMIC_RELAXED);
                    buckets[pos - 1]      = graph.n() + i;
                }
            });
        }
    );
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

        for (NodeID c_u = r.begin(); c_u != r.end(); ++c_u) {
            edge_buffer.mark(c_u);

            const std::size_t first_pos = buckets_position_buffer[c_u];
            const std::size_t last_pos  = buckets_position_buffer[c_u + 1];

            auto collect_edges = [&](auto& map) {
                NodeWeight c_u_weight = 0;

                for (std::size_t i = first_pos; i < last_pos; ++i) {
                    const NodeID u = buckets[i];

                    auto handle_edge = [&](const EdgeWeight weight, const GlobalNodeID cluster) {
                        if (graph.is_owned_global_node(cluster)) {
                            const NodeID c_local_node = cluster_mapping[cluster - graph.offset_n()];
                            if (c_local_node != c_u) {
                                map[c_local_node] += weight;
                            }
                        } else {
                            auto& handle = nonlocal_cluster_filter_handle_ets.local();
                            auto  it     = handle.find(cluster + 1);
                            DBG << "Edge to ghost cluster " << cluster << ": "
                                << (it != handle.end() ? "FOUND" : "NOT-FOUND");
                            if (it != handle.end() && (*it).second - 1 < graph.global_n()) {
                                const std::size_t  index           = (*it).second - 1;
                                const PEID         owner           = graph.find_owner_of_global_node(cluster);
                                const GlobalNodeID c_ghost_node    = their_mapping_responses[owner][index];
                                auto               c_local_node_it = c_global_to_ghost.find(c_ghost_node + 1);
                                const NodeID       c_local_node    = (*c_local_node_it).second;

                                DBG << " --> index " << index << ", owner " << owner << ", ghost node " << c_ghost_node
                                    << " and local node " << c_local_node << " (from " << c_u << ") --> "
                                    << (c_local_node != c_u ? "ADD" : "REJECT-SELF-LOOP");

                                if (c_local_node != c_u) {
                                    map[c_local_node] += weight;
                                }
                            }
                        }
                    };

                    if (u < graph.n()) {
                        c_u_weight += graph.node_weight(u);
                        for (const auto [e, v]: graph.neighbors(u)) {
                            handle_edge(graph.edge_weight(e), clustering[v]);
                            /*const NodeID c_v = mapping[v];
                            DBG << "Edge " << u << " (" << c_u << ") --> " << v << " (" << c_v << ")";
                            if (c_v != c_u) {
                                map[c_v] += graph.edge_weight(e);
                            }*/
                        }
                    } else {
                        // Fix node weight later
                        for (std::size_t index = u - graph.n(); local_edges[index].u == c_u; ++index) {
                            handle_edge(local_edges[index].weight, local_edges[index].v); //@todo
                        }
                    }
                }

                c_node_weights[c_u] = c_u_weight;
                c_nodes[c_u + 1]    = map.size();

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
    STOP_TIMER();

    tbb::parallel_for<std::size_t>(0, local_nodes.size(), [&](const std::size_t i) {
        const NodeID c_u = cluster_mapping[local_nodes[i].u - graph.offset_n()];
        __atomic_fetch_add(&c_node_weights[c_u], local_nodes[i].weight, __ATOMIC_RELAXED);
    });

    parallel::prefix_sum(c_nodes.begin(), c_nodes.end(), c_nodes.begin());

    // Build edge distribution
    START_TIMER("Build coarse edge distribution");
    const EdgeID c_m = c_nodes.back();
    DBG << "Number of coarse edges: " << c_m;

    scalable_vector<GlobalEdgeID> c_edge_distribution(size + 1);
    MPI_Allgather(
        &c_m, 1, mpi::type::get<EdgeID>(), c_edge_distribution.data(), 1, mpi::type::get<GlobalEdgeID>(),
        graph.communicator()
    );
    std::exclusive_scan(c_edge_distribution.begin(), c_edge_distribution.end(), c_edge_distribution.begin(), 0u);
    DBG << "Coarse edge distribution: [" << c_edge_distribution << "]";
    STOP_TIMER();

    auto all_buffered_nodes = ts_navigable_list::combine<NodeID, LocalEdge, scalable_vector>(edge_buffer_ets);

    START_TIMER("Allocation");
    scalable_vector<NodeID>     c_edges(c_m);
    scalable_vector<EdgeWeight> c_edge_weights(c_m);
    STOP_TIMER();

    // Finally, build coarse graph
    START_TIMER("Construct coarse graph");
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

    return {std::move(c_graph), std::move(mapping)};
}
} // namespace kaminpar::dist
