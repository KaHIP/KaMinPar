/*******************************************************************************
 * @file:   graph_permutation.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Computes graph permutations and builds the permuted graph.
 ******************************************************************************/
#pragma once

#include "common/parallel/algorithm.h"
#include "common/parallel/loops.h"
#include "kaminpar/context.h"
#include "kaminpar/datastructure/graph.h"
#include "kaminpar/definitions.h"

namespace kaminpar::graph {
/*!
 * Bidirectional node permutation.
 * @tparam Container
 */
template <template <typename> typename Container>
struct NodePermutations {
    Container<NodeID> old_to_new;
    Container<NodeID> new_to_old;
};

/*!
 * Computes a permutation on the nodes of the graph such that nodes are sorted by their exponentially spaced degree
 * buckets. Isolated nodes moved to the back of the graph.
 *
 * @tparam Container
 * @param nodes Nodes array of a static graph.
 * @return Bidirectional node permutation.
 */
template <template <typename> typename Container, bool put_deg0_at_end = true>
NodePermutations<Container> sort_by_degree_buckets(const Container<EdgeID>& nodes) {
    auto find_bucket = [&](const Degree deg) {
        return deg == 0 ? (put_deg0_at_end ? kNumberOfDegreeBuckets - 1 : 0) : degree_bucket(deg);
    };

    const NodeID n    = nodes.size() - 1;
    const int    cpus = std::min<int>(tbb::this_task_arena::max_concurrency(), n);

    Container<NodeID> permutation(n);
    Container<NodeID> inverse_permutation(n);

    // local_buckets[cpu][bucket]: thread-local bucket sizes
    using Buckets = std::array<NodeID, kNumberOfDegreeBuckets + 1>;
    std::vector<Buckets, tbb::cache_aligned_allocator<Buckets>> local_buckets(cpus + 1);

    parallel::deterministic_for<NodeID>(0, n, [&](const NodeID from, const NodeID to, const int cpu) {
        KASSERT(cpu < cpus);
        for (NodeID u = from; u < to; ++u) {
            const auto bucket = find_bucket(nodes[u + 1] - nodes[u]);
            permutation[u]    = local_buckets[cpu + 1][bucket]++;
        }
    });

    // Build a table of prefix numbers to correct the position of each node in the final permutation
    // After the previous loop, permutation[u] contains the position of u in the thread-local bucket.
    // (i) account for smaller buckets --> add prefix computed in global_buckets
    // (ii) account for the same bucket in smaller processor IDs --> add prefix computed in local_buckets
    Buckets global_buckets{};
    for (int id = 1; id < cpus + 1; ++id) {
        for (std::size_t i = 0; i + 1 < global_buckets.size(); ++i) {
            global_buckets[i + 1] += local_buckets[id][i];
        }
    }
    parallel::prefix_sum(global_buckets.begin(), global_buckets.end(), global_buckets.begin());
    for (std::size_t i = 0; i < global_buckets.size(); ++i) {
        for (int id = 0; id + 1 < cpus; ++id) {
            local_buckets[id + 1][i] += local_buckets[id][i];
        }
    }

    // Apply offsets to obtain global permutation
    parallel::deterministic_for<NodeID>(0, n, [&](const NodeID from, const NodeID to, const int cpu) {
        KASSERT(cpu < cpus);

        for (NodeID u = from; u < to; ++u) {
            const Degree bucket = find_bucket(nodes[u + 1] - nodes[u]);
            permutation[u] += global_buckets[bucket] + local_buckets[cpu][bucket];
        }
    });

    // Compute inverse permutation
    tbb::parallel_for(static_cast<std::size_t>(1), nodes.size(), [&](const NodeID u_plus_one) {
        const NodeID u                      = u_plus_one - 1;
        inverse_permutation[permutation[u]] = u;
    });

    return {std::move(permutation), std::move(inverse_permutation)};
}

/*!
 * Creates a permuted copy of a graph.
 * @tparam Container
 * @tparam has_ghost_nodes If true, edge targets may not exist. These are not permuted.
 * @param old_nodes Original nodes array of a static graph.
 * @param old_edges Original edges array of a static graph.
 * @param old_node_weights Original node weights, may be empty.
 * @param old_edge_weights Original edge weights, may be empty.
 * @param permutations Node permutation.
 * @param new_nodes New nodes array, must already be allocated.
 * @param new_edges New edges array, must already be allocated.
 * @param new_node_weights New node weights, may be empty iff. the old node weights array is empty.
 * @param new_edge_weights New edge weights, may be empty empty iff. the old edge weights array is empty.
 */
template <template <typename> typename Container, bool has_ghost_nodes = false>
void build_permuted_graph(
    const Container<EdgeID>& old_nodes, const Container<NodeID>& old_edges,
    const Container<NodeWeight>& old_node_weights, const Container<EdgeWeight>& old_edge_weights,
    const NodePermutations<Container>& permutations, Container<EdgeID>& new_nodes, Container<NodeID>& new_edges,
    Container<NodeWeight>& new_node_weights, Container<EdgeWeight>& new_edge_weights) {
    // >= for ghost nodes in a distributed graph
    const bool is_node_weighted = old_node_weights.size() + 1 >= old_nodes.size();
    const bool is_edge_weighted = old_edge_weights.size() == old_edges.size();

    const NodeID n = old_nodes.size() - 1;
    KASSERT(n + 1 == new_nodes.size());

    // Build p_nodes, p_node_weights
    tbb::parallel_for<NodeID>(0, n, [&](const NodeID u) {
        const NodeID old_u = permutations.new_to_old[u];

        new_nodes[u] = old_nodes[old_u + 1] - old_nodes[old_u];
        if (is_node_weighted) {
            new_node_weights[u] = old_node_weights[old_u];
        }
    });
    parallel::prefix_sum(new_nodes.begin(), new_nodes.end(), new_nodes.begin());

    // Build p_edges, p_edge_weights
    tbb::parallel_for(static_cast<NodeID>(0), n, [&](const NodeID u) {
        const NodeID old_u = permutations.new_to_old[u];

        for (EdgeID e = old_nodes[old_u]; e < old_nodes[old_u + 1]; ++e) {
            const NodeID v   = old_edges[e];
            const EdgeID p_e = --new_nodes[u];
            new_edges[p_e]   = (!has_ghost_nodes || v < n) ? permutations.old_to_new[v] : v;
            if (is_edge_weighted) {
                new_edge_weights[p_e] = old_edge_weights[e];
            }
        }
    });
}
} // namespace kaminpar::graph
