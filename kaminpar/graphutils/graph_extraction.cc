/*******************************************************************************
 * @file:   graph_extraction.cc
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Extracts the subgraphs induced by each block of a partition.
 ******************************************************************************/
#include "kaminpar/graphutils/graph_extraction.h"
#include <mutex>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar/datastructure/graph.h"
#include "kaminpar/datastructure/static_array.h"
#include "kaminpar/definitions.h"
#include "kaminpar/parallel/algorithm.h"
#include "kaminpar/parallel/atomic.h"
#include "kaminpar/utils/math.h"
#include "kaminpar/utils/timer.h"

#include "kaminpar/graphutils/graph_extraction.tcc"

namespace kaminpar::graph {
SequentialSubgraphExtractionResult extract_subgraphs_sequential(
    const PartitionedGraph& p_graph, const SubgraphMemoryStartPosition memory_position, SubgraphMemory& subgraph_memory,
    TemporarySubgraphMemory& tmp_subgraph_memory) {
    ALWAYS_ASSERT(p_graph.k() == 2) << "Only suitable for bipartitions!";
    ALWAYS_ASSERT(tmp_subgraph_memory.in_use == false);
    tmp_subgraph_memory.in_use = true;

    const bool is_node_weighted = p_graph.graph().is_node_weighted();
    const bool is_edge_weighted = p_graph.graph().is_edge_weighted();

    const BlockID final_k = p_graph.final_k(0) + p_graph.final_k(1);
    tmp_subgraph_memory.ensure_size_nodes(p_graph.n() + final_k, is_node_weighted);

    auto& nodes        = tmp_subgraph_memory.nodes;
    auto& edges        = tmp_subgraph_memory.edges;
    auto& node_weights = tmp_subgraph_memory.node_weights;
    auto& edge_weights = tmp_subgraph_memory.edge_weights;
    auto& mapping      = tmp_subgraph_memory.mapping;

    std::array<NodeID, 2> s_n{0, 0};
    std::array<EdgeID, 2> s_m{0, 0};

    // find graph sizes
    for (const NodeID u: p_graph.nodes()) {
        const BlockID b                = p_graph.block(u);
        tmp_subgraph_memory.mapping[u] = s_n[b]++;

        for (const auto [e, v]: p_graph.neighbors(u)) {
            if (p_graph.block(v) == b) {
                ++s_m[b];
            }
        }
    }

    // start position of subgraph[1] in common memory ds
    const NodeID n1 = s_n[0] + p_graph.final_k(0);
    const EdgeID m1 = s_m[0];

    nodes[0]  = 0;
    nodes[n1] = 0;
    tmp_subgraph_memory.ensure_size_edges(s_m[0] + s_m[1], is_edge_weighted);

    // build extract graphs in temporary memory buffer
    std::array<EdgeID, 2> next_edge_id{0, 0};

    for (const NodeID u: p_graph.nodes()) {
        const BlockID b = p_graph.block(u);

        const NodeID n0 = b * n1; // either 0 or s_n[0] + final_k(0)
        const EdgeID m0 = b * m1; // either 0 or s_m[0]

        for (const auto [e, v]: p_graph.neighbors(u)) {
            if (p_graph.block(v) == b) {
                edges[m0 + next_edge_id[b]] = mapping[v];
                if (is_edge_weighted) {
                    edge_weights[m0 + next_edge_id[b]] = p_graph.edge_weight(e);
                }
                ++next_edge_id[b];
            }
        }

        nodes[n0 + mapping[u] + 1] = next_edge_id[b];
        if (is_node_weighted) {
            node_weights[n0 + mapping[u]] = p_graph.node_weight(u);
        }
    }

    // copy graphs to subgraph_memory at memory_position
    // THIS OPERATION OVERWRITES p_graph!
    std::copy(
        nodes.begin(), nodes.begin() + p_graph.n() + final_k,
        subgraph_memory.nodes.begin() + memory_position.nodes_start_pos);
    std::copy(
        edges.begin(), edges.begin() + s_m[0] + s_m[1],
        subgraph_memory.edges.begin() + memory_position.edges_start_pos);
    if (is_node_weighted) {
        std::copy(
            node_weights.begin(), node_weights.begin() + p_graph.n() + final_k,
            subgraph_memory.node_weights.begin() + memory_position.nodes_start_pos);
    }
    if (is_edge_weighted) {
        std::copy(
            edge_weights.begin(), edge_weights.begin() + s_m[0] + s_m[1],
            subgraph_memory.edge_weights.begin() + memory_position.edges_start_pos);
    }

    tmp_subgraph_memory.in_use = false;

    std::array<SubgraphMemoryStartPosition, 2> subgraph_positions;
    subgraph_positions[0].nodes_start_pos = memory_position.nodes_start_pos;
    subgraph_positions[0].edges_start_pos = memory_position.edges_start_pos;
    subgraph_positions[1].nodes_start_pos = memory_position.nodes_start_pos + n1;
    subgraph_positions[1].edges_start_pos = memory_position.edges_start_pos + m1;

    auto create_graph = [&](const NodeID n0, const NodeID n, const EdgeID m0, const EdgeID m) {
        StaticArray<EdgeID>     s_nodes(memory_position.nodes_start_pos + n0, n + 1, subgraph_memory.nodes);
        StaticArray<NodeID>     s_edges(memory_position.edges_start_pos + m0, m, subgraph_memory.edges);
        StaticArray<NodeWeight> s_node_weights(
            is_node_weighted * (memory_position.nodes_start_pos + n0), is_node_weighted * n,
            subgraph_memory.node_weights);
        StaticArray<EdgeWeight> s_edge_weights(
            is_edge_weighted * (memory_position.edges_start_pos + m0), is_edge_weighted * m,
            subgraph_memory.edge_weights);
        return Graph{
            tag::seq, std::move(s_nodes), std::move(s_edges), std::move(s_node_weights), std::move(s_edge_weights)};
    };

    std::array<Graph, 2> subgraphs{create_graph(0, s_n[0], 0, s_m[0]), create_graph(n1, s_n[1], m1, s_m[1])};

    return {std::move(subgraphs), std::move(subgraph_positions)};
}

/*
 * Builds a block-induced subgraph for each block of a partitioned graph. Return type contains a mapping that maps
 * nodes from p_graph to nodes in the respective subgraph; we need this because the order in which nodes in subgraphs
 * appear is non-deterministic due to parallelization.
 */
SubgraphExtractionResult extract_subgraphs(const PartitionedGraph& p_graph, SubgraphMemory& subgraph_memory) {
    return extract_subgraphs_impl(p_graph, subgraph_memory, [&](const BlockID b) { return p_graph.final_k(b); });
}

namespace {
void fill_final_k(scalable_vector<BlockID>& data, const BlockID b0, const BlockID final_k, const BlockID k) {
    const auto [final_k1, final_k2] = math::split_integral(final_k);
    std::array<BlockID, 2> ks{
        std::clamp<BlockID>(std::ceil(k * 1.0 * final_k1 / final_k), 1, k - 1),
        std::clamp<BlockID>(std::floor(k * 1.0 * final_k2 / final_k), 1, k - 1)};
    std::array<BlockID, 2> b{b0, b0 + ks[0]};
    data[b[0]] = final_k1;
    data[b[1]] = final_k2;

    if (ks[0] > 1) {
        fill_final_k(data, b[0], final_k1, ks[0]);
    }
    if (ks[1] > 1) {
        fill_final_k(data, b[1], final_k2, ks[1]);
    }
}
} // namespace

void copy_subgraph_partitions(
    PartitionedGraph& p_graph, const scalable_vector<BlockArray>& p_subgraph_partitions, const BlockID k_prime,
    const BlockID input_k, const scalable_vector<NodeID>& mapping) {
    scalable_vector<BlockID> k0(p_graph.k() + 1, k_prime / p_graph.k());
    k0[0] = 0;

    scalable_vector<BlockID> final_ks(k_prime, 1);

    // we are done partitioning? --> use final_ks
    if (k_prime == input_k) {
        std::copy(p_graph.final_ks().begin(), p_graph.final_ks().end(), k0.begin() + 1);
    }

    parallel::prefix_sum(k0.begin(), k0.end(), k0.begin()); // blocks of old block i start at k0[i]

    // we are not done partitioning?
    if (k_prime != input_k) {
        ALWAYS_ASSERT(math::is_power_of_2(k_prime));
        const BlockID k_per_block = k_prime / p_graph.k();
        tbb::parallel_for(static_cast<BlockID>(0), p_graph.k(), [&](const BlockID b) {
            fill_final_k(final_ks, k0[b], p_graph.final_k(b), k_per_block);
        });
    }

    p_graph.change_k(k_prime);
    tbb::parallel_for(static_cast<NodeID>(0), p_graph.n(), [&](const NodeID& u) {
        const BlockID b   = p_graph.block(u);
        const NodeID  s_u = mapping[u];
        p_graph.set_block<false>(u, k0[b] + p_subgraph_partitions[b][s_u]);
    });

    p_graph.set_final_ks(std::move(final_ks));
    p_graph.reinit_block_weights();
}
} // namespace kaminpar::graph
