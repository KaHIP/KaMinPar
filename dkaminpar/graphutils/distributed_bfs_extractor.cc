/*******************************************************************************
 * @file:   distributed_bfs_extractor.cc
 * @author: Daniel Seemaier
 * @date:   31.08.2022
 * @brief:  Algorithm to extract a region of the graph using a BFS search.
 ******************************************************************************/
#include "dkaminpar/graphutils/distributed_bfs_extractor.h"

#include "dkaminpar/datastructure/distributed_graph.h"

#include "common/parallel/algorithm.h"

namespace kaminpar::dist {
DistributedBfsExtractor::DistributedBfsExtractor(const DistributedGraph& graph) : _graph(graph) {}

void DistributedBfsExtractor::initialize(const DistributedPartitionedGraph* p_graph) {
    _p_graph = p_graph;
}

DistributedBfsExtractor::Result DistributedBfsExtractor::extract(const std::vector<NodeID>& seed_nodes) {
    return {};
}

std::size_t DistributedBfsExtractor::compute_serialized_subgraph_size(const NoinitVector<std::pair<NodeID, bool>>& nodes
) {
    const NodeID num_nodes = nodes.size();
    const EdgeID num_edges = parallel::accumulate(nodes.begin(), nodes.end(), 0u, [this](const auto& node_border_pair) {
        const auto& [node, is_border_node] = node_border_pair;
        return is_border_node ? _p_graph->k() : _graph.degree(node);
    });
    return compute_serialization_offsets<NodeID>(num_nodes, num_edges).total_size;
}

void DistributedBfsExtractor::serialize_subgraph(
    const NoinitVector<std::pair<NodeID, bool>>& nodes, PreallocatedVector<NodeID>& storage
) {
    const NodeID num_nodes = nodes.size();
    const EdgeID num_edges = parallel::accumulate(nodes.begin(), nodes.end(), 0u, [this](const auto& node_border_pair) {
        const auto& [node, is_border_node] = node_border_pair;
        return is_border_node ? _p_graph->k() : _graph.degree(node);
    });

    const auto    offsets          = compute_serialization_offsets<NodeID>(num_nodes, num_edges);
    NodeID*       num_nodes_ptr    = storage.data();
    EdgeID*       nodes_ptr        = reinterpret_cast<EdgeID*>(storage.data() + offsets.nodes);
    NodeID*       edges_ptr        = storage.data() + offsets.edges;
    NodeWeight*   node_weights_ptr = reinterpret_cast<NodeWeight*>(storage.data() + offsets.node_weights);
    EdgeWeight*   edge_weights_ptr = reinterpret_cast<EdgeWeight*>(storage.data() + offsets.edge_weights);
    GlobalNodeID* node_mapping_ptr = reinterpret_cast<GlobalNodeID*>(storage.data() + offsets.node_mapping);

    *num_nodes_ptr = num_nodes;

    // Copy degree, node weight, mapping
    tbb::parallel_for<std::size_t>(0, num_nodes, [&](const std::size_t i) {
        const auto& [node, is_border_node] = nodes[i];
        const EdgeID degree                = is_border_node ? _p_graph->k() : _graph.degree(node);
        *(nodes_ptr + i)                   = degree;
        *(node_weights_ptr + i)            = _graph.node_weight(node);
        *(node_mapping_ptr + i)            = _graph.local_to_global_node(node);
    });

    // Compute almost-nodes array (actual entry + degree)
    parallel::prefix_sum(nodes_ptr, nodes_ptr + num_nodes, nodes_ptr);
    *(nodes_ptr + num_nodes) = num_edges;

    // Copy edges
    tbb::parallel_for<std::size_t>(0, num_nodes, [&](const std::size_t i) {
        const auto& [node, is_border_node] = nodes[i];
        EdgeID& edge_pos                   = *(nodes_ptr + i);

        if (is_border_node) {
            // @todo
        } else {
            for (const auto& [edge, neighbor]: _graph.neighbors(node)) {
                --edge_pos;
                *(edges_ptr + edge_pos)        = _graph.local_to_global_node(neighbor);
                *(edge_weights_ptr + edge_pos) = _graph.edge_weight(edge);
            }
        }
    });
}

template <typename UnderlyingType>
DistributedBfsExtractor::SerializationOffsets
DistributedBfsExtractor::compute_serialization_offsets(const NodeID num_nodes, const EdgeID num_edges) {
    const std::size_t node_id_multiplier        = sizeof(NodeID) / sizeof(UnderlyingType);
    const std::size_t global_node_id_multiplier = sizeof(GlobalNodeID) / sizeof(UnderlyingType);
    const std::size_t edge_id_multiplier        = sizeof(EdgeID) / sizeof(UnderlyingType);
    const std::size_t node_weight_multiplier    = sizeof(NodeWeight) / sizeof(UnderlyingType);
    const std::size_t edge_weight_multiplier    = sizeof(EdgeWeight) / sizeof(UnderlyingType);

    SerializationOffsets offsets;
    offsets.nodes        = node_id_multiplier;
    offsets.edges        = offsets.nodes + edge_id_multiplier * (num_nodes + 1);
    offsets.node_weights = offsets.edges + node_id_multiplier * num_edges;
    offsets.edge_weights = offsets.node_weights + node_weight_multiplier * num_nodes;
    offsets.node_mapping = offsets.edge_weights + node_weight_multiplier * num_edges;
    offsets.total_size   = offsets.node_mapping + global_node_id_multiplier * num_nodes;
    return offsets;
}

DistributedBfsExtractor::DeserializedGraph
DistributedBfsExtractor::deserialize_subgraph(NoinitVector<NodeID>& serialized_graph) {
    return {};
}
} // namespace kaminpar::dist
