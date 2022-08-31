/*******************************************************************************
 * @file:   bfs_extractor.cc
 * @author: Daniel Seemaier
 * @date:   23.08.2022
 * @brief:  Algorithm to extract a region of the graph using a BFS search.
 ******************************************************************************/
#include "dkaminpar/graphutils/bfs_extractor.h"

#include <algorithm>
#include <random>
#include <stack>
#include <vector>

#include <kassert/kassert.hpp>
#include <tbb/parallel_for.h>

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/growt.h"
#include "dkaminpar/mpi/alltoall.h"

#include "common/datastructures/marker.h"
#include "common/random.h"

namespace kaminpar::dist::graph {
SET_DEBUG(true);

BfsExtractor::BfsExtractor(const DistributedGraph& graph) : _graph(&graph) {}

void BfsExtractor::initialize(const DistributedPartitionedGraph& p_graph) {
    _p_graph = &p_graph;
}

auto BfsExtractor::extract(const std::vector<NodeID>& seed_nodes) -> Result {
    // Initialize external degrees if needed
    if (_exterior_strategy == ExteriorStrategy::CONTRACT && _external_degrees.empty()) {
        init_external_degrees();
    }

    const PEID size = mpi::get_comm_size(_graph->communicator());
    const PEID rank = mpi::get_comm_rank(_graph->communicator());

    NoinitVector<GhostSeedNode> initial_seed_nodes;
    for (const auto seed_node: seed_nodes) {
        initial_seed_nodes.emplace_back(0, seed_node);
    }

    auto local_bfs_result = bfs(initial_seed_nodes, {});

    std::vector<NoinitVector<GhostSeedEdge>> cur_ghost_seed_edges(size);
    cur_ghost_seed_edges[rank] = std::move(local_bfs_result.explored_ghosts);

    for (PEID hop = 0; hop < _max_hops; ++hop) {
        auto  ghost_exchange_result = exchange_ghost_seed_nodes(cur_ghost_seed_edges);
        auto& next_ghost_seed_nodes = ghost_exchange_result.first;
        auto& next_ignored_nodes    = ghost_exchange_result.second;

        tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
            auto continued_bfs_result = bfs(next_ghost_seed_nodes[pe], next_ignored_nodes[pe]);
            cur_ghost_seed_edges[pe]  = std::move(continued_bfs_result.explored_ghosts);
        });
    }
}

auto BfsExtractor::exchange_ghost_seed_nodes(std::vector<NoinitVector<GhostSeedEdge>>& outgoing_ghost_seed_edges)
    -> std::pair<std::vector<NoinitVector<GhostSeedNode>>, std::vector<NoinitVector<NodeID>>> {
    const PEID size = mpi::get_comm_size(_graph->communicator());

    // Exchange new ghost nodes
    std::vector<NoinitVector<NodeID>> sendbufs(size);
    for (PEID pe = 0; pe < size; ++pe) {
        // Ghost seed nodes to continue the BFS search initiated on PE `pe`
        const auto& ghost_seed_edges = outgoing_ghost_seed_edges[pe];

        for (const auto& [local_interface_node, local_ghost_node, distance]: ghost_seed_edges) {
            KASSERT(_graph->is_ghost_node(local_ghost_node));

            const auto ghost_owner       = _graph->ghost_owner(local_ghost_node);
            const auto global_ghost_node = _graph->local_to_global_node(local_ghost_node);
            const auto remote_ghost_node =
                asserting_cast<NodeID>(global_ghost_node - _graph->node_distribution(ghost_owner));

            sendbufs[ghost_owner].push_back(static_cast<NodeID>(pe));
            sendbufs[ghost_owner].push_back(local_interface_node);
            sendbufs[ghost_owner].push_back(remote_ghost_node);
            sendbufs[ghost_owner].push_back(distance);
        }
    }

    std::vector<NoinitVector<GhostSeedEdge>> incoming_ghost_seed_edges(size);
    mpi::sparse_alltoall<NodeID>(
        std::move(sendbufs),
        [&](const auto& recvbuf, const PEID pe) {
            for (std::size_t i = 0; i < recvbuf.size();) {
                const PEID   initiating_pe        = asserting_cast<PEID>(recvbuf[i++]);
                const NodeID remote_ghost_node    = recvbuf[i++];
                const NodeID local_interface_node = recvbuf[i++];
                const NodeID distance             = recvbuf[i++];

                const auto global_ghost_node = _graph->node_distribution(pe) + remote_ghost_node;
                const auto local_ghost_node  = _graph->global_to_local_node(global_ghost_node);

                incoming_ghost_seed_edges[initiating_pe].emplace_back(local_ghost_node, local_interface_node, distance);
            }
        },
        _graph->communicator()
    );

    // Filter edges that where already explored from this PE
    std::vector<NoinitVector<GhostSeedNode>> next_ghost_seed_nodes(size);
    std::vector<NoinitVector<NodeID>>        next_ignored_nodes(size);

    tbb::parallel_for<PEID>(0, size, [&](const PEID pe) {
        auto& outgoing_edges = outgoing_ghost_seed_edges[pe];
        auto& incoming_edges = incoming_ghost_seed_edges[pe];
        std::sort(outgoing_edges.begin(), outgoing_edges.end());
        std::sort(incoming_edges.begin(), incoming_edges.end());

        // std::size_t outgoing_pos  = 0;
        for (std::size_t incoming_pos = 0; incoming_pos < incoming_edges.size(); ++incoming_pos) {
            const auto& [cur_ghost_node, cur_interface_node, distance] = incoming_edges[incoming_pos];

            // @todo filter
            // Only use this as a ghost node

            next_ghost_seed_nodes[pe].emplace_back(distance, cur_interface_node);
            next_ignored_nodes[pe].push_back(cur_ghost_node);
        }
    });

    return {std::move(next_ghost_seed_nodes), std::move(next_ignored_nodes)};
}

/*std::vector<BfsExtractor::GraphSegment> exchange_subgraphs(const std::vector<std::vector<NodeID>>& nodes) {
    const PEID size = mpi::get_comm_size(_graph->communicator());

    // Build subgraph segments for each PE
    std::vector<std::vector<NodeID>> sendbufs(static_cast<std::size_t>(size));
    for (PEID pe = 0; pe < size; ++pe) {
    }
}*/

/*BfsExtractedGraph BfsExtractor::extract_from_node(const NodeID start_node) {
    return {};
}*/

auto BfsExtractor::bfs(NoinitVector<GhostSeedNode>& ghost_seed_nodes, const NoinitVector<NodeID>& ignored_nodes)
    -> ExploredSubgraph {
    // Catch case where there are no seed nodes
    if (ghost_seed_nodes.empty()) {
        return {};
    }

    // Maximum number of nodes to be explored by this BFS search
    const NodeID max_size = _graph->total_n() < _max_size ? _graph->total_n() : _max_size * ghost_seed_nodes.size();

    // Marker for nodes that were already explored by this BFS search
    auto& taken = _taken_ets.local();
    taken.reset();
    for (const NodeID ignored_node: ignored_nodes) {
        // Prevent that we explore edges from which this BFS continues from another PE
        taken.set(ignored_node);
    }

    // Sort ghost seed nodes by distance
    std::sort(ghost_seed_nodes.begin(), ghost_seed_nodes.end(), [](const auto& lhs, const auto& rhs) {
        return std::get<0>(lhs) < std::get<0>(rhs);
    });

    // Initialize search from closest ghost seed nodes
    NodeID                      current_distance = std::get<0>(ghost_seed_nodes.front());
    std::stack<NodeID>          front;
    NoinitVector<ExploredNode>  visited_nodes;
    NoinitVector<GhostSeedEdge> next_ghost_seed_edges;

    auto add_seed_nodes_to_search_front = [&](const NodeID with_distance) {
        for (const auto& [distance, node]: ghost_seed_nodes) {
            if (distance == with_distance) {
                front.push(node);
            } else if (with_distance < distance) {
                break;
            }
        }
    };

    // Initialize search front
    add_seed_nodes_to_search_front(current_distance);
    NodeID front_size = front.size();
    add_seed_nodes_to_search_front(current_distance + 1);

    // Perform BFS
    while (!front.empty() && visited_nodes.size() < _max_size && current_distance < _max_radius) {
        const NodeID node = front.top();
        front.pop();
        --front_size;

        // Explore neighbors of owned nodes
        if (_graph->is_owned_node(node)) {
            explore_outgoing_edges(node, [&](const NodeID neighbor) {
                if (taken.get(neighbor)) {
                    // Skip nodes already discovered
                    return true;
                }

                if (!_graph->is_owned_node(neighbor)) {
                    // Record ghost seeds for the next hop
                    next_ghost_seed_edges.emplace_back(node, neighbor, current_distance + 1);
                    // do not mark as taken
                    return true;
                }

                if (current_distance + 1 < _max_radius) {
                    // Add to stack for the next search layer
                    front.push(neighbor);
                }

                // Record node as discovered
                const bool is_border_node = current_distance + 1 == _max_radius;
                visited_nodes.emplace_back(neighbor, is_border_node);
                taken.set(neighbor);

                return true;
            });
        }

        if (front_size == 0) {
            front_size = front.size();
            ++current_distance;
            add_seed_nodes_to_search_front(current_distance + 1);
        }
    }

    // @todo if _max_size aborted BFS, mark nodes without neighbors as border nodes

    ExploredSubgraph ans;
    ans.explored_nodes  = std::move(visited_nodes);
    ans.explored_ghosts = std::move(next_ghost_seed_edges);
    return ans;
}

template <typename Lambda>
void BfsExtractor::explore_outgoing_edges(const NodeID node, Lambda&& lambda) {
    const bool is_high_degree_node = _graph->degree(node) >= _high_degree_threshold;

    if (!is_high_degree_node || _high_degree_strategy == HighDegreeStrategy::TAKE_ALL) {
        for (const auto [e, v]: _graph->neighbors(node)) {
            if (!lambda(v)) {
                break;
            }
        }
    } else if (_high_degree_strategy == HighDegreeStrategy::CUT) {
        for (EdgeID e = _graph->first_edge(node); e < _graph->first_edge(node) + _high_degree_threshold; ++e) {
            if (!lambda(_graph->edge_target(e))) {
                break;
            }
        }
    } else if (_high_degree_strategy == HighDegreeStrategy::SAMPLE) {
        const double                        skip_prob = 1.0 * _high_degree_threshold / _graph->degree(node);
        std::geometric_distribution<EdgeID> skip_dist(skip_prob);

        for (EdgeID e = _graph->first_edge(node); e < _graph->first_invalid_edge(node);
             ++e) { // e += skip_dist(gen)) { // @todo
            if (!lambda(_graph->edge_target(e))) {
                break;
            }
        }
    } else {
        // do nothing for HighDegreeStrategy::IGNORE
    }
}

BfsExtractor::InducedSubgraph BfsExtractor::build_induced_subgraph(const std::vector<NodeID>& nodes) {
    // Build mappings from and to subgraph
    const auto&                        mapping_subgraph_to_graph = nodes;
    std::unordered_map<NodeID, NodeID> mapping_graph_to_subgraph;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        mapping_graph_to_subgraph[nodes[i]] = asserting_cast<NodeID>(i);
    }

    // Data strucutes for subgraph
    const NodeID            sub_n       = nodes.size() + _p_graph->k();
    const NodeID            sub_n_prime = nodes.size();
    std::vector<EdgeID>     sub_nodes(sub_n + 1);
    std::vector<NodeID>     sub_edges;
    std::vector<NodeWeight> sub_node_weights(sub_n);
    std::vector<EdgeWeight> sub_edge_weights;

    std::vector<EdgeWeight> edge_weights_to_block_nodes(_p_graph->k());

    for (const NodeID u: nodes) {
        sub_nodes[u]        = asserting_cast<EdgeID>(sub_edges.size());
        sub_node_weights[u] = _graph->node_weight(u);

        for (const auto [e, v]: _graph->neighbors(u)) {
            auto v_it = mapping_graph_to_subgraph.find(v);

            if (v_it != mapping_graph_to_subgraph.end()) {
                // v is also in the subgraph
                const auto [v_prime, sub_v] = *v_it;
                KASSERT(v_prime == v);
                sub_edges.push_back(sub_v);
                sub_edge_weights.push_back(_graph->edge_weight(e));
            } else {
                // v is not in the subgraph -> contributes to our edge to a block node
                edge_weights_to_block_nodes[_p_graph->block(v)] += _graph->edge_weight(e);
            }
        }

        for (const BlockID b: _p_graph->blocks()) {
            if (edge_weights_to_block_nodes[b] > 0) {
                sub_edges.push_back(sub_n_prime + b);
                sub_edge_weights.push_back(edge_weights_to_block_nodes[b]);
                edge_weights_to_block_nodes[b] = 0;
            }
        }
    }

    for (NodeID u = nodes.size(); u < sub_nodes.size(); ++u) {
        sub_nodes[u] = sub_edges.size();
    }

    // Copy subgraph to StaticArray<>
    shm::Graph subgraph = [&] {
        StaticArray<EdgeID>     real_sub_nodes(sub_nodes.size());
        StaticArray<NodeID>     real_sub_edges(sub_edges.size());
        StaticArray<NodeWeight> real_sub_node_weights(sub_node_weights.size());
        StaticArray<EdgeWeight> real_sub_edge_weights(sub_edge_weights.size());
        std::copy(sub_nodes.begin(), sub_nodes.end(), real_sub_nodes.begin());
        std::copy(sub_edges.begin(), sub_edges.end(), real_sub_edges.begin());
        std::copy(sub_node_weights.begin(), sub_node_weights.end(), real_sub_node_weights.begin());
        std::copy(sub_edge_weights.begin(), sub_edge_weights.end(), real_sub_edge_weights.begin());
        return shm::Graph(
            std::move(real_sub_nodes), std::move(real_sub_edges), std::move(real_sub_node_weights),
            std::move(real_sub_edge_weights)
        );
    }();

    NoinitVector<GlobalNodeID> mapping(mapping_subgraph_to_graph.size());
    std::transform(
        mapping_subgraph_to_graph.begin(), mapping_subgraph_to_graph.end(), mapping.begin(),
        [&](const NodeID node) { return _graph->local_to_global_node(node); }
    );

    NoinitVector<BlockID> partition(sub_n);
    std::transform(
        mapping_subgraph_to_graph.begin(), mapping_subgraph_to_graph.end(), partition.begin(),
        [&](const NodeID node) { return _p_graph->block(node); }
    );

    return {std::move(subgraph), std::move(partition), std::move(mapping)};
}

auto BfsExtractor::build_result(std::vector<IncompleteSubgraph>& fragments) -> Result {
    return {};
}

void BfsExtractor::set_max_hops(const PEID max_hops) {
    _max_hops = max_hops;
}

void BfsExtractor::set_max_radius(const GlobalNodeID max_radius) {
    _max_radius = max_radius;
}

void BfsExtractor::set_max_size(const GlobalNodeID max_size) {
    _max_size = max_size;
}

void BfsExtractor::set_high_degree_threshold(const GlobalEdgeID high_degree_threshold) {
    _high_degree_threshold = high_degree_threshold;
}

void BfsExtractor::set_high_degree_strategy(const HighDegreeStrategy high_degree_strategy) {
    _high_degree_strategy = high_degree_strategy;
}

void BfsExtractor::init_external_degrees() {
    _external_degrees.resize(_graph->n() * _p_graph->k());
    tbb::parallel_for<std::size_t>(0, _external_degrees.size(), [&](const std::size_t i) { _external_degrees[i] = 0; });

    _graph->pfor_nodes([&](const NodeID u) {
        for (const auto [e, v]: _graph->neighbors(u)) {
            const BlockID    v_block  = _p_graph->block(v);
            const EdgeWeight e_weight = _graph->edge_weight(e);
            external_degree(u, v_block) += e_weight;
        }
    });
}

EdgeWeight& BfsExtractor::external_degree(const NodeID u, const BlockID b) {
    KASSERT(u * _p_graph->k() + b < _external_degrees.size());
    return _external_degrees[u * _p_graph->k() + b];
}
} // namespace kaminpar::dist::graph

