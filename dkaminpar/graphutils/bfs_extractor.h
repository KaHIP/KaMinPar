/*******************************************************************************
 * @file:   bfs_extractor.h
 * @author: Daniel Seemaier
 * @date:   23.08.2022
 * @brief:  Algorithm to extract a region of the graph using a BFS search.
 ******************************************************************************/
#pragma once

#include <limits>
#include <type_traits>

#include <tbb/enumerable_thread_specific.h>

#include "datastructures/fast_reset_array.h"

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/definitions.h"

#include "kaminpar/datastructure/graph.h"

#include "common/datastructures/marker.h"
#include "common/noinit_vector.h"

namespace kaminpar::dist::graph {
class BfsExtractor {
public:
    struct Result {
        std::unique_ptr<shm::Graph>            graph;
        std::unique_ptr<shm::PartitionedGraph> p_graph;
        NoinitVector<GlobalNodeID>             node_mapping;
    };

    enum HighDegreeStrategy { IGNORE, TAKE_ALL, SAMPLE, CUT };

    enum ExteriorStrategy { EXCLUDE, INCLUDE, CONTRACT };

    BfsExtractor(const DistributedGraph& graph);

    BfsExtractor& operator=(const BfsExtractor&) = delete;
    BfsExtractor& operator=(BfsExtractor&&)      = delete;
    BfsExtractor(const BfsExtractor&)            = delete;
    BfsExtractor(BfsExtractor&&) noexcept        = default;

    void initialize(const DistributedPartitionedGraph& p_graph);

    Result extract(const std::vector<NodeID>& seed_nodes);

    void set_max_hops(PEID max_hops);
    void set_max_radius(GlobalNodeID max_radius);
    void set_max_size(GlobalNodeID max_size);
    void set_high_degree_threshold(GlobalEdgeID high_degree_threshold);
    void set_high_degree_strategy(HighDegreeStrategy high_degree_strategy);
    void set_exterior_strategy(ExteriorStrategy exterior_strategy);

private:
    using GhostSeedNode = std::pair<NodeID, NodeID>;          // distance, node
    using GhostSeedEdge = std::tuple<NodeID, NodeID, NodeID>; // distance, from, to
    using ExploredNode  = std::pair<NodeID, bool>;            // is border node, node

    struct ExploredSubgraph {
        NoinitVector<ExploredNode>  explored_nodes;
        NoinitVector<GhostSeedEdge> explored_ghosts;
    };

    ExploredSubgraph bfs(NoinitVector<GhostSeedNode>& seed_nodes, const NoinitVector<NodeID>& ignored_nodes);

    struct IncompleteSubgraph {
        NoinitVector<EdgeID>       nodes;
        NoinitVector<GlobalNodeID> edges;
        NoinitVector<NodeWeight>   node_weights;
        NoinitVector<EdgeWeight>   edge_weights;
        NoinitVector<GlobalNodeID> node_mapping;
    };

    template <typename Lambda>
    void explore_outgoing_edges(NodeID node, Lambda&& action);

    std::pair<std::vector<NoinitVector<GhostSeedNode>>, std::vector<NoinitVector<NodeID>>>
    exchange_ghost_seed_nodes(std::vector<NoinitVector<GhostSeedEdge>>& next_ghost_seed_nodes);

    struct GraphSegment {
        std::vector<EdgeID>       nodes;
        std::vector<GlobalNodeID> edges;
        std::vector<NodeWeight>   node_weights;
        std::vector<EdgeWeight>   edge_weights;
    };

//    std::vector<GraphSegment> exchange_subgraphs(const std::vector<std::vector<NodeID>>& nodes);

    struct InducedSubgraph {
        shm::Graph                 graph;
        NoinitVector<BlockID>      partition;
        NoinitVector<GlobalNodeID> mapping;
    };

    InducedSubgraph build_induced_subgraph(const std::vector<NodeID>& nodes);

    Result build_result(std::vector<IncompleteSubgraph> &fragments);

    void        init_external_degrees();
    EdgeWeight& external_degree(NodeID u, BlockID b);

    const DistributedGraph*            _graph   = nullptr;
    const DistributedPartitionedGraph* _p_graph = nullptr;

    PEID               _max_hops{std::numeric_limits<PEID>::max()};
    GlobalNodeID       _max_radius{std::numeric_limits<GlobalNodeID>::max()};
    GlobalNodeID       _max_size{std::numeric_limits<GlobalNodeID>::max()};
    GlobalEdgeID       _high_degree_threshold{std::numeric_limits<GlobalEdgeID>::max()};
    HighDegreeStrategy _high_degree_strategy{HighDegreeStrategy::TAKE_ALL};
    ExteriorStrategy   _exterior_strategy{ExteriorStrategy::EXCLUDE};

    NoinitVector<EdgeWeight> _external_degrees{};

    Marker<> _finished_pe_search{static_cast<std::size_t>(mpi::get_comm_size(_graph->communicator()))};

    tbb::enumerable_thread_specific<Marker<>> _taken_ets{[&] {
        return Marker<>(_graph->total_n());
    }};
};
} // namespace kaminpar::dist::graph

