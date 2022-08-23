/*******************************************************************************
 * @file:   bfs_extractor.h
 * @author: Daniel Seemaier
 * @date:   23.08.2022
 * @brief:  Algorithm to extract a region of the graph using a BFS search.
 ******************************************************************************/
#pragma once

#include <limits>
#include <type_traits>

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/definitions.h"

#include "kaminpar/datastructure/graph.h"

namespace kaminpar::dist::graph {
class BfsExtractedGraph {
public:
    shm::Graph& graph() {
        return _graph;
    }

    shm::PartitionedGraph& p_graph() {
        // The pointer to _graph in _p_graph might dangle if this object was moved.
        // Thus, always update the pointer before returning the reference.
        _p_graph.update_graph_ptr(&_graph);
        return _p_graph;
    }

private:
    shm::Graph            _graph;
    shm::PartitionedGraph _p_graph;
};

class BfsExtractor {
public:
    enum HighDegreeStrategy { IGNORE, TAKE_ALL, SAMPLE, CUT };
    enum ExteriorStrategy { EXCLUDE, INCLUDE, CONTRACT };

    BfsExtractor(const DistributedPartitionedGraph& p_graph);

    std::vector<BfsExtractedGraph> extract(const std::vector<NodeID>& start_nodes);
    BfsExtractedGraph              extract(NodeID start_node);

    void allow_overlap();
    void set_max_hops(PEID max_hops);
    void set_max_radius(GlobalNodeID max_radius);
    void set_max_size(GlobalNodeID max_size);
    void set_high_degree_threshold(GlobalEdgeID high_degree_threshold);
    void set_high_degree_strategy(HighDegreeStrategy high_degree_strategy);
    void set_exterior_strategy(ExteriorStrategy exterior_strategy);

private:
    void grow_multi_seeded_region(const std::vector<std::vector<NodeID>>& seed_nodes_per_distance);

    void        init_external_degrees();
    EdgeWeight& external_degree(NodeID u, BlockID b);

    const DistributedPartitionedGraph& _p_graph;

    bool               _overlap{false};
    PEID               _max_hops{std::numeric_limits<PEID>::max()};
    GlobalNodeID       _max_radius{std::numeric_limits<GlobalNodeID>::max()};
    GlobalNodeID       _max_size{std::numeric_limits<GlobalNodeID>::max()};
    GlobalEdgeID       _high_degree_threshold{std::numeric_limits<GlobalEdgeID>::max()};
    HighDegreeStrategy _high_degree_strategy{HighDegreeStrategy::TAKE_ALL};
    ExteriorStrategy   _exterior_strategy{ExteriorStrategy::EXCLUDE};

    NoinitVector<EdgeWeight> _external_degrees{};
};
} // namespace kaminpar::dist::graph

