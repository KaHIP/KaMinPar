/*******************************************************************************
 * @file:   graph_extraction.cc
 *
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 * @brief:  Distributes block-induced subgraphs of a partitioned graph across
 * PEs.
 ******************************************************************************/
#include "dkaminpar/graphutils/graph_extraction.h"

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/mpi_graph.h"
#include "dkaminpar/utils/math.h"
#include "dkaminpar/utils/vector_ets.h"
#include "kaminpar/parallel/algorithm.h"

#include <functional>

namespace dkaminpar::graph {
namespace {
PEID compute_block_owner(const BlockID b, const BlockID k, const PEID num_pes) {
    return static_cast<PEID>(math::compute_local_range_rank<BlockID>(k, static_cast<BlockID>(num_pes), b));
}

auto count_block_induced_subgraph_sizes(const DistributedPartitionedGraph& p_graph) {
    parallel::vector_ets<NodeID> num_nodes_per_block_ets(p_graph.k());
    parallel::vector_ets<EdgeID> num_edges_per_block_ets(p_graph.k());

    tbb::parallel_for(tbb::blocked_range<NodeID>(0, p_graph.n()), [&](const auto r) {
        auto& num_nodes_per_block = num_nodes_per_block_ets.local();
        auto& num_edges_per_block = num_edges_per_block_ets.local();
        for (NodeID u = r.begin(); u != r.end(); ++u) {
            const BlockID u_block = p_graph.block(u);
            ++num_nodes_per_block[u_block];
            for (const auto [e, v]: p_graph.neighbors(u)) {
                if (u_block == p_graph.block(v)) {
                    ++num_edges_per_block[u_block];
                }
            }
        }
    });

    return std::make_pair(num_nodes_per_block_ets.combine(std::plus{}), num_edges_per_block_ets.combine(std::plus{}));
}

struct ExtractedSubgraphs {
    std::vector<EdgeID>     shared_nodes;
    std::vector<NodeWeight> shared_node_weights;
    std::vector<NodeID>     shared_edges;
    std::vector<EdgeWeight> shared_edge_weights;

    std::vector<std::size_t> nodes_offset;
    std::vector<std::size_t> edges_offset;

    std::vector<NodeID> mapping;
};

// Build a local block-induced subgraph for each block of the graph partition.
ExtractedSubgraphs
extract_local_block_induced_subgraphs(const DistributedPartitionedGraph& p_graph, ExtractedSubgraphs memory = {}) {
    auto [num_nodes_per_block, num_edges_per_block] = count_block_induced_subgraph_sizes(p_graph);
    const EdgeID num_internal_edges = std::accumulate(num_edges_per_block.begin(), num_edges_per_block.end(), 0);

    auto& shared_nodes          = memory.shared_nodes;
    auto& shared_node_weights   = memory.shared_node_weights;
    auto& shared_edges          = memory.shared_edges;
    auto& shared_edge_weights   = memory.shared_edge_weights;
    auto& nodes_offset          = memory.nodes_offset;
    auto& edges_offset          = memory.edges_offset;
    auto& mapping               = memory.mapping;
    auto  next_node_in_subgraph = std::vector<parallel::Atomic<NodeID>>();

    // Allocate memory
    {
        SCOPED_TIMER("Allocation", TIMER_DETAIL);

        const std::size_t min_nodes_size   = p_graph.n();
        const std::size_t min_edges_size   = num_internal_edges;
        const std::size_t min_offset_size  = p_graph.k() + 1;
        const std::size_t min_mapping_size = p_graph.total_n();

        LIGHT_ASSERT(shared_nodes.size() == shared_node_weights.size());
        LIGHT_ASSERT(shared_edges.size() == shared_edge_weights.size());
        LIGHT_ASSERT(nodes_offset.size() == edges_offset.size());

        if (shared_nodes.size() < min_nodes_size) {
            shared_nodes.resize(min_nodes_size);
            shared_node_weights.resize(min_nodes_size);
        }
        if (shared_edges.size() < min_edges_size) {
            shared_edges.resize(min_edges_size);
            shared_edge_weights.resize(min_edges_size);
        }
        if (nodes_offset.size() < min_offset_size) {
            nodes_offset.resize(min_offset_size);
            edges_offset.resize(min_offset_size);
        }
        if (mapping.size() < min_mapping_size) {
            mapping.resize(min_mapping_size);
        }

        next_node_in_subgraph.resize(p_graph.k());
    }

    // Compute of graphs in shared_* arrays
    {
        SCOPED_TIMER("Compute subgraph offsets", TIMER_DETAIL);

        parallel::prefix_sum(num_nodes_per_block.begin(), num_nodes_per_block.end(), nodes_offset.begin() + 1);
        parallel::prefix_sum(num_edges_per_block.begin(), num_edges_per_block.end(), edges_offset.begin() + 1);
    }

    // Build mapping from node IDs in p_graph to node IDs in the extracted subgraph
    {
        SCOPED_TIMER("Build node mapping", TIMER_DETAIL);

        // @todo bottleneck for scalibility
        p_graph.pfor_nodes([&](const NodeID u) {
            const BlockID b               = p_graph.block(u);
            const NodeID  pos_in_subgraph = next_node_in_subgraph[b]++;
            const NodeID  pos             = nodes_offset[u] + pos_in_subgraph;
            shared_nodes[pos]             = u;
            mapping[u]                    = pos_in_subgraph;
        });
    }

    // Exchange mapping for ghost nodes
    {
        SCOPED_TIMER("Exchange ghost node mapping", TIMER_DETAIL);

        struct NodeToMappedNode {
            GlobalNodeID global_node;
            NodeID       mapped_node;
        };

        mpi::graph::sparse_alltoall_interface_to_pe<NodeToMappedNode>(
            p_graph.graph(),
            [&](const NodeID u) {
                return NodeToMappedNode{p_graph.local_to_global_node(u), mapping[u]};
            },
            [&](const auto buffer) {
                tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
                    const auto& [global_node, mapped_node] = buffer[i];
                    const NodeID local_node                = p_graph.global_to_local_node(global_node);
                    mapping[local_node]                    = mapped_node;
                });
            });
    }

    // Extract the subgraphs
    {
        SCOPED_TIMER("Extract subgraphs", TIMER_DETAIL);

        tbb::parallel_for<BlockID>(0, p_graph.k(), [&](const BlockID b) {
            const NodeID n0 = nodes_offset[b];
            const EdgeID e0 = edges_offset[b];
            EdgeID       e  = 0;

            // u, v, e = IDs in extracted subgraph
            // u_prime, v_prime, e_prime = IDs in p_graph
            for (NodeID u = 0; u < next_node_in_subgraph[b]; ++u) {
                const NodeID pos     = n0 + u;
                const NodeID u_prime = shared_nodes[pos];

                for (const auto [e_prime, v_prime]: p_graph.neighbors(u_prime)) {
                    if (p_graph.block(v_prime) != b) {
                        continue;
                    }

                    shared_edge_weights[e0 + e] = p_graph.edge_weight(e_prime);
                    shared_edges[e0 + e]        = mapping[v_prime];
                    ++e;
                }

                shared_nodes[pos]        = e;
                shared_node_weights[pos] = p_graph.node_weight(u_prime);
            }
        });
    }

    return memory;
}
} // namespace

std::vector<DistributedGraph> distribute_block_induced_subgraphs(const DistributedPartitionedGraph& p_graph) {
    return {};
}
} // namespace dkaminpar::graph
