/*******************************************************************************
 * @file:   distributed_bfs_extractor.h
 * @author: Daniel Seemaier
 * @date:   31.08.2022
 * @brief:  Algorithm to extract a region of the graph using a BFS search.
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/definitions.h"

#include "kaminpar/datastructure/graph.h"

#include "common/noinit_vector.h"
#include "common/preallocated_vector.h"

namespace kaminpar::dist {
class DistributedBfsExtractor {
public:
    struct Result {
        std::unique_ptr<shm::Graph>            graph;
        std::unique_ptr<shm::PartitionedGraph> p_graph;
        NoinitVector<GlobalNodeID>             node_mapping;
    };

    DistributedBfsExtractor(const DistributedGraph& graph);

    void initialize(const DistributedPartitionedGraph* p_graph);

    Result extract(const std::vector<NodeID>& seed_nodes);

private:
    std::size_t compute_serialized_subgraph_size(const NoinitVector<std::pair<NodeID, bool>>& nodes);

    void serialize_subgraph(const NoinitVector<std::pair<NodeID, bool>>& nodes, PreallocatedVector<NodeID>& storage);

    struct DeserializedGraph {
        NodeID                           num_nodes;
        EdgeID                           num_edges;
        PreallocatedVector<EdgeID>       nodes;
        PreallocatedVector<NodeID>       edges;
        PreallocatedVector<NodeWeight>   node_weights;
        PreallocatedVector<EdgeWeight>   edge_weights;
        PreallocatedVector<GlobalNodeID> node_mapping;
    };

    DeserializedGraph deserialize_subgraph(NoinitVector<NodeID> &serialized_graph);

    struct SerializationOffsets {
        std::size_t nodes;
        std::size_t edges;
        std::size_t node_weights;
        std::size_t edge_weights;
        std::size_t node_mapping;
        std::size_t total_size;
    };

    template <typename UnderlyingType>
    SerializationOffsets compute_serialization_offsets(const NodeID num_nodes, const EdgeID num_edges);

    const DistributedGraph&            _graph;
    const DistributedPartitionedGraph* _p_graph;
};
} // namespace kaminpar::dist
