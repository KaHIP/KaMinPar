/*******************************************************************************
 * @file:   graph_extraction.tcc
 *
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 * @brief:  Extracts the subgraphs induced by each block of a partition.
 ******************************************************************************/
#include "kaminpar/datastructure/graph.h"

#include "kaminpar/definitions.h"
#include "kaminpar/graphutils/graph_extraction.h"
#include "kaminpar/parallel/algorithm.h"
#include "kaminpar/utils/timer.h"

namespace kaminpar::graph {
/*
 * Builds a block-induced subgraph for each block of a partitioned graph. Return type contains a mapping that maps
 * nodes from p_graph to nodes in the respective subgraph; we need this because the order in which nodes in subgraphs
 * appear is non-deterministic due to parallelization.
 */
template <typename PartitionedGraphType, typename NodeOffsetCB>
SubgraphExtractionResult extract_subgraphs_impl(
    const PartitionedGraphType& p_graph, SubgraphMemory& subgraph_memory, const NodeOffsetCB&& node_offset_cb) {
    const auto& graph = p_graph.graph();

    START_TIMER("Allocation");
    scalable_vector<NodeID>                      mapping(p_graph.n());
    scalable_vector<SubgraphMemoryStartPosition> start_positions(p_graph.k() + 1);
    std::vector<parallel::Atomic<NodeID>>        bucket_index(p_graph.n());
    scalable_vector<Graph>                       subgraphs(p_graph.k());
    STOP_TIMER();

    // count number of nodes and edges in each block
    START_TIMER("Count block size");
    tbb::enumerable_thread_specific<scalable_vector<NodeID>> tl_num_nodes_in_block{[&] {
        return scalable_vector<NodeID>(p_graph.k());
    }};
    tbb::enumerable_thread_specific<scalable_vector<EdgeID>> tl_num_edges_in_block{[&] {
        return scalable_vector<EdgeID>(p_graph.k());
    }};

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, graph.n()), [&](auto& r) {
        auto& num_nodes_in_block = tl_num_nodes_in_block.local();
        auto& num_edges_in_block = tl_num_edges_in_block.local();

        for (NodeID u = r.begin(); u != r.end(); ++u) {
            const BlockID u_block = p_graph.block(u);
            ++num_nodes_in_block[u_block];
            for (const NodeID v: graph.adjacent_nodes(u)) {
                if (p_graph.block(v) == u_block) {
                    ++num_edges_in_block[u_block];
                }
            }
        }
    });
    STOP_TIMER();

    START_TIMER("Merge block sizes");
    tbb::parallel_for(static_cast<BlockID>(0), p_graph.k(), [&](const BlockID b) {
        NodeID num_nodes = node_offset_cb(b); // padding for sequential subgraph extraction
        EdgeID num_edges = 0;
        for (auto& local_num_nodes: tl_num_nodes_in_block) {
            num_nodes += local_num_nodes[b];
        }
        for (auto& local_num_edges: tl_num_edges_in_block) {
            num_edges += local_num_edges[b];
        }
        start_positions[b + 1].nodes_start_pos = num_nodes;
        start_positions[b + 1].edges_start_pos = num_edges;
    });
    parallel::prefix_sum(start_positions.begin(), start_positions.end(), start_positions.begin());
    STOP_TIMER();

    // build temporary bucket array in nodes array
    START_TIMER("Build bucket array");
    tbb::parallel_for(static_cast<NodeID>(0), p_graph.n(), [&](const NodeID u) {
        const BlockID b               = p_graph.block(u);
        const NodeID  pos_in_subgraph = bucket_index[b]++;
        const NodeID  pos             = start_positions[b].nodes_start_pos + pos_in_subgraph;
        subgraph_memory.nodes[pos]    = u;
        mapping[u]                    = pos_in_subgraph; // concurrent random access write
    });
    STOP_TIMER();

    const bool is_node_weighted = p_graph.graph().is_node_weighted();
    const bool is_edge_weighted = p_graph.graph().is_edge_weighted();

    // build graph
    START_TIMER("Construct subgraphs");
    tbb::parallel_for(static_cast<BlockID>(0), p_graph.k(), [&](const BlockID b) {
        const NodeID nodes_start_pos = start_positions[b].nodes_start_pos;
        EdgeID       e               = 0;              // edge = in subgraph
        for (NodeID u = 0; u < bucket_index[b]; ++u) { // u = in subgraph
            const NodeID pos           = nodes_start_pos + u;
            const NodeID u_prime       = subgraph_memory.nodes[pos]; // u_prime = in graph
            subgraph_memory.nodes[pos] = e;
            if (is_node_weighted) {
                subgraph_memory.node_weights[pos] = graph.node_weight(u_prime);
            }

            const EdgeID e0 = start_positions[b].edges_start_pos;

            for (const auto [e_prime, v_prime]: graph.neighbors(u_prime)) { // e_prime, v_prime = in graph
                if (p_graph.block(v_prime) == b) {                          // only keep internal edges
                    if (is_edge_weighted) {
                        subgraph_memory.edge_weights[e0 + e] = graph.edge_weight(e_prime);
                    }
                    subgraph_memory.edges[e0 + e] = mapping[v_prime];
                    ++e;
                }
            }
        }

        subgraph_memory.nodes[nodes_start_pos + bucket_index[b]] = e;
    });
    STOP_TIMER();

    START_TIMER("Create graph objects");
    tbb::parallel_for(static_cast<BlockID>(0), p_graph.k(), [&](const BlockID b) {
        const NodeID n0 = start_positions[b].nodes_start_pos;
        const EdgeID m0 = start_positions[b].edges_start_pos;
        const NodeID n  = start_positions[b + 1].nodes_start_pos - n0 - node_offset_cb(b);
        const EdgeID m  = start_positions[b + 1].edges_start_pos - m0;

        StaticArray<EdgeID>     nodes(n0, n + 1, subgraph_memory.nodes);
        StaticArray<NodeID>     edges(m0, m, subgraph_memory.edges);
        StaticArray<NodeWeight> node_weights(is_node_weighted * n0, is_node_weighted * n, subgraph_memory.node_weights);
        StaticArray<EdgeWeight> edge_weights(is_edge_weighted * m0, is_edge_weighted * m, subgraph_memory.edge_weights);
        subgraphs[b] = Graph{std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights)};
    });
    STOP_TIMER();

    HEAVY_ASSERT([&] {
        for (const BlockID b: p_graph.blocks()) {
            LOG << "Validate " << b;
            ALWAYS_ASSERT(validate_graph(subgraphs[b]));
        }
        return true;
    });

    return {std::move(subgraphs), std::move(mapping), std::move(start_positions)};
}
} // namespace kaminpar::graph
