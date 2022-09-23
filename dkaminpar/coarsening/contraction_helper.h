/*******************************************************************************
 * @file:   contraction_helper.h
 * @author: Daniel Seemaier
 * @date:   29.10.2021
 * @brief:  Utility functions for contracting distributed graphs.
 ******************************************************************************/
#pragma once

#include <algorithm>

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dkaminpar/definitions.h"

#include "common/datastructures/ts_navigable_linked_list.h"
#include "common/noinit_vector.h"
#include "common/scalable_vector.h"
#include "common/timer.h"
#include "common/utils/math.h"

#include <oneapi/tbb/parallel_sort.h>
#include <tbb/parallel_sort.h>

namespace kaminpar::dist::helper {
namespace {
SET_DEBUG(false);
}

struct LocalToGlobalEdge {
    NodeID       u;
    EdgeWeight   weight;
    GlobalNodeID v;
};

struct DeduplicateEdgeListMemoryContext {
    NoinitVector<parallel::Atomic<NodeID>> bucket_index;
    NoinitVector<NodeID>                   deduplicated_bucket_index;
    NoinitVector<LocalToGlobalEdge>        buffer_list;
};

template <typename Container>
inline Container
deduplicate_edge_list2(Container edge_list) {
    START_TIMER("Sorting edges");
    tbb::parallel_sort(edge_list.begin(), edge_list.end(), [&](const auto &lhs, const auto &rhs) {
        return lhs.u < rhs.u || (lhs.u == rhs.u && lhs.v < rhs.v);
    });
    STOP_TIMER();

    START_TIMER("Compressing edges");
    std::size_t free = 0; // @todo parallelize
    for (std::size_t i = 0; i < edge_list.size();) {
        edge_list[free] = edge_list[i];
        auto &acc_edge = edge_list[free - 1];

        ++free;
        ++i;

        while (acc_edge.u == edge_list[i].u && acc_edge.v == edge_list[i].v) {
           acc_edge.weight += edge_list[i].weight;
           ++i;
        }
    }
    STOP_TIMER();

    edge_list.resize(free);

    return edge_list;
}

template <typename Container>
inline std::pair<Container, DeduplicateEdgeListMemoryContext>
deduplicate_edge_list(Container edge_list, const NodeID n, DeduplicateEdgeListMemoryContext m_ctx) {
    SCOPED_TIMER("Deduplicate edge list", TIMER_DETAIL);

    auto& bucket_index              = m_ctx.bucket_index;
    auto& deduplicated_bucket_index = m_ctx.deduplicated_bucket_index;
    auto& buffer_list               = m_ctx.buffer_list;

    TIMED_SCOPE("Allocation", TIMER_DETAIL) {
        if (bucket_index.size() < n + 1) {
            bucket_index.resize(n + 1);
        }
        if (deduplicated_bucket_index.size() < n + 1) {
            deduplicated_bucket_index.resize(n + 1);
        }
        if (buffer_list.size() < edge_list.size() + 1) {
            buffer_list.resize(edge_list.size() + 1);
        }
    };

    // sort edges by u and store result in compressed_edge_list
    TIMED_SCOPE("Bucket sort edges", TIMER_DETAIL) {
        tbb::parallel_for<NodeID>(0, n + 1, [&](const NodeID u) { bucket_index[u] = 0; });
        tbb::parallel_for<std::size_t>(0, edge_list.size(), [&](const std::size_t i) {
            KASSERT(edge_list[i].u < n);
            bucket_index[edge_list[i].u].fetch_add(1, std::memory_order_relaxed);
        });
        parallel::prefix_sum(bucket_index.begin(), bucket_index.end(), bucket_index.begin());
        tbb::parallel_for<std::size_t>(0, edge_list.size(), [&](const std::size_t i) {
            const std::size_t j = bucket_index[edge_list[i].u].fetch_sub(1, std::memory_order_relaxed) - 1;
            buffer_list[j]      = edge_list[i];
        });
        buffer_list.back().v = kInvalidGlobalNodeID; // dummy element
    };

    // sort outgoing edges for each node and collapse duplicated edges
    START_TIMER("Deduplicate edges", TIMER_DETAIL);

    START_TIMER("Count degrees", TIMER_DETAIL);
    tbb::parallel_for<NodeID>(0, n, [&](const NodeID u) {
        const EdgeID first_edge_id         = bucket_index[u];
        const EdgeID first_invalid_edge_id = bucket_index[u + 1];

        // sort outgoing edges from u
        std::sort(
            buffer_list.begin() + first_edge_id, buffer_list.begin() + first_invalid_edge_id,
            [&](const auto& lhs, const auto& rhs) { return lhs.v < rhs.v; }
        );

        // compute degree of u after deduplicating
        EdgeID       deduplicated_degree = 0;
        GlobalNodeID current_v           = kInvalidGlobalNodeID;

        for (EdgeID edge_id = first_edge_id; edge_id < first_invalid_edge_id; ++edge_id) {
            KASSERT(buffer_list[edge_id].u == u);
            const auto& edge = buffer_list[edge_id];
            if (edge.v != current_v) {
                ++deduplicated_degree;
                current_v = edge.v;
            }
        }

        deduplicated_bucket_index[u + 1] = deduplicated_degree;
    });
    STOP_TIMER(TIMER_DETAIL);

    START_TIMER("Compute prefix sum over degrees", TIMER_DETAIL);
    deduplicated_bucket_index[0] = 0;
    parallel::prefix_sum(
        deduplicated_bucket_index.begin(), deduplicated_bucket_index.begin() + n + 1, deduplicated_bucket_index.begin()
    );
    STOP_TIMER();

    // now copy edges to edge_list
    START_TIMER("Copy edges to compressed edge list", TIMER_DETAIL);
    tbb::parallel_for<NodeID>(0, n, [&](const NodeID u) {
        const EdgeID first_edge_id         = bucket_index[u];
        const EdgeID first_invalid_edge_id = bucket_index[u + 1];
        std::size_t  dst_index             = deduplicated_bucket_index[u];
        GlobalNodeID current_v             = kInvalidGlobalNodeID;

        for (EdgeID edge_id = first_edge_id; edge_id < first_invalid_edge_id; ++edge_id) {
            const auto& edge = buffer_list[edge_id];
            if (edge.v == current_v) {
                KASSERT(edge_list[dst_index - 1].u == edge.u);
                KASSERT(edge_list[dst_index - 1].v == edge.v);
                edge_list[dst_index - 1].weight += edge.weight;
            } else {
                current_v              = edge.v;
                edge_list[dst_index++] = edge;
            }
        }
    });
    STOP_TIMER(TIMER_DETAIL);
    STOP_TIMER(TIMER_DETAIL);

    START_TIMER("Resize edge list", TIMER_DETAIL);
    edge_list.resize(deduplicated_bucket_index[n]);
    STOP_TIMER(TIMER_DETAIL);

    return {std::move(edge_list), std::move(m_ctx)};
}

template <typename T>
inline scalable_vector<T> create_perfect_distribution_from_global_count(const T global_count, MPI_Comm comm) {
    const auto size = mpi::get_comm_size(comm);

    scalable_vector<T> distribution(size + 1);
    for (PEID pe = 0; pe < size; ++pe) {
        distribution[pe + 1] = math::compute_local_range<T>(global_count, size, pe).second;
    }

    return distribution;
}

template <typename T>
inline scalable_vector<T> create_distribution_from_local_count(const T local_count, MPI_Comm comm) {
    const auto [size, rank] = mpi::get_comm_info(comm);

    scalable_vector<T> distribution(size + 1);
    mpi::allgather(&local_count, 1, distribution.data() + 1, 1, comm);
    parallel::prefix_sum(distribution.begin(), distribution.end(), distribution.begin());
    distribution.front() = 0;

    return distribution;
}

/**
 * Constructs a distributed graph from an edge list.
 * @param edge_list List of edges with the following fields: \c u, \c v and \c weight, where \c u is a local node ID and
 * \c v is a global node ID.
 * @return Distributed graph built from the edge list.
 */
template <typename EdgeList, typename NodeWeightLambda, typename FindGhostNodeOwnerLambda>
inline DistributedGraph build_distributed_graph_from_edge_list(
    const EdgeList& edge_list, scalable_vector<GlobalNodeID> node_distribution, MPI_Comm comm,
    NodeWeightLambda&& node_weight_lambda, FindGhostNodeOwnerLambda&& /* find_ghost_node_owner */
) {
    SCOPED_TIMER("Build graph from edge list", TIMER_DETAIL);

    const PEID   rank = mpi::get_comm_rank(comm);
    const NodeID n    = node_distribution[rank + 1] - node_distribution[rank];

    // Bucket-sort edge list
    START_TIMER("Bucket sort edge list", TIMER_DETAIL);
    scalable_vector<parallel::Atomic<NodeID>> bucket_index(n + 1);
    tbb::parallel_for<std::size_t>(0, edge_list.size(), [&](const std::size_t i) {
        bucket_index[edge_list[i].u].fetch_add(1, std::memory_order_relaxed);
    });
    parallel::prefix_sum(bucket_index.begin(), bucket_index.end(), bucket_index.begin());
    scalable_vector<std::size_t> buckets(edge_list.size());
    tbb::parallel_for<std::size_t>(0, edge_list.size(), [&](const std::size_t i) {
        buckets[bucket_index[edge_list[i].u].fetch_sub(1, std::memory_order_relaxed) - 1] = i;
    });
    STOP_TIMER(TIMER_DETAIL);

    // Assertion:
    // Buckets of node u go from bucket_index[u]..bucket_index[u + 1]
    // edge_list[buckets[bucket_index[u] + i]] is the i-th outgoing edge from node u

    // Construct the edges of the graph in thread-local buffers
    struct Edge {
        GlobalNodeID v;
        EdgeWeight   weight;
    };
    NavigableLinkedList<NodeID, Edge, scalable_vector> edge_buffer_ets;

    START_TIMER("Allocation", TIMER_DETAIL);
    scalable_vector<EdgeID> nodes(n + 1);
    STOP_TIMER(TIMER_DETAIL);

    START_TIMER("Build coarse edges", TIMER_DETAIL);
    tbb::parallel_for(tbb::blocked_range<NodeID>(0, n), [&](const auto r) {
        auto& edge_buffer = edge_buffer_ets.local();

        for (NodeID u = r.begin(); u != r.end(); ++u) {
            edge_buffer.mark(u);

            const std::size_t u_bucket_start = bucket_index[u];
            const std::size_t u_bucket_end   = bucket_index[u + 1];

            // Sort outgoing edges from u by target node
            std::sort(
                buckets.begin() + u_bucket_start, buckets.begin() + u_bucket_end,
                [&](const auto& lhs, const auto& rhs) { return edge_list[lhs].v < edge_list[rhs].v; }
            );

            // Construct outgoing edges
            EdgeID       degree         = 0;
            GlobalNodeID current_v      = kInvalidGlobalNodeID;
            EdgeWeight   current_weight = 0;

            for (std::size_t i = u_bucket_start; i < u_bucket_end; ++i) {
                const GlobalNodeID v      = edge_list[buckets[i]].v;
                const EdgeWeight   weight = edge_list[buckets[i]].weight;

                if (v != current_v) {
                    if (current_v != kInvalidGlobalNodeID) {
                        edge_buffer.push_back({current_v, current_weight});
                        ++degree;
                    }

                    current_v      = v;
                    current_weight = 0;
                }

                current_weight += weight;
            }

            if (current_v != kInvalidGlobalNodeID) { // finish last edge if there was at least one edge
                edge_buffer.push_back({current_v, current_weight});
                ++degree;
            }

            nodes[u + 1] = degree;
        }
    });

    parallel::prefix_sum(nodes.begin(), nodes.end(), nodes.begin());
    const auto all_buffered_nodes = ts_navigable_list::combine<NodeID, Edge, scalable_vector>(edge_buffer_ets);
    STOP_TIMER(TIMER_DETAIL);

    START_TIMER("Allocation", TIMER_DETAIL);
    const EdgeID                m = nodes.back();
    scalable_vector<NodeID>     edges(m);
    scalable_vector<EdgeWeight> edge_weights(m);
    STOP_TIMER(TIMER_DETAIL);

    const GlobalNodeID from = node_distribution[rank];
    const GlobalNodeID to   = node_distribution[rank + 1];

    // Now construct the coarse graph
    START_TIMER("Construct coarse graph", TIMER_DETAIL);
    graph::GhostNodeMapper mapper{node_distribution, comm};

    DBG << "Number of nodes n=" << n;
    EdgeID num_entries_used       = 0;
    EdgeID num_ghost_entries_used = 0;

    tbb::parallel_for<NodeID>(0, n, [&](const NodeID i) {
        const auto&  marker = all_buffered_nodes[i];
        const auto*  list   = marker.local_list;
        const NodeID u      = marker.key;

        // Copy edges to edges + edge weights arrays
        const EdgeID u_degree        = nodes[u + 1] - nodes[u];
        const EdgeID first_src_index = marker.position;
        const EdgeID first_dst_index = nodes[u];

        for (EdgeID j = 0; j < u_degree; ++j) {
            const auto dst_index = first_dst_index + j;
            const auto src_index = first_src_index + j;

            const auto [v, weight] = list->get(src_index);

            if (from <= v && v < to) {
                edges[dst_index] = static_cast<NodeID>(v - from);
                ++num_entries_used;
            } else {
                edges[dst_index] = mapper.new_ghost_node(v);
                ++num_ghost_entries_used;
            }
            edge_weights[dst_index] = weight;
        }
    });
    DBG << V(num_entries_used) << V(num_ghost_entries_used);
    STOP_TIMER(TIMER_DETAIL);

    START_TIMER("Finalize coarse graph mapping", TIMER_DETAIL);
    auto [global_to_ghost, ghost_to_global, ghost_owner] = mapper.finalize();
    const NodeID ghost_n                                 = ghost_to_global.size();
    STOP_TIMER(TIMER_DETAIL);

    // node weights for ghost nodes must be computed afterwards
    START_TIMER("Construct coarse node weights", TIMER_DETAIL);
    scalable_vector<NodeWeight> node_weights(n + ghost_n);
    tbb::parallel_for<NodeID>(0, n, [&](const NodeID u) { node_weights[u] = node_weight_lambda(u); });
    STOP_TIMER(TIMER_DETAIL);

    return {
        std::move(node_distribution),
        create_distribution_from_local_count<GlobalEdgeID>(m, comm),
        std::move(nodes),
        std::move(edges),
        std::move(node_weights),
        std::move(edge_weights),
        std::move(ghost_owner),
        std::move(ghost_to_global),
        std::move(global_to_ghost),
        false,
        comm};
}
} // namespace kaminpar::dist::helper
