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

BfsExtractor::BfsExtractor(const DistributedPartitionedGraph& p_graph) : _p_graph(p_graph) {}

std::vector<BfsExtractedGraph> BfsExtractor::extract(const std::vector<NodeID>& start_nodes) {
    // Initialize external degrees if needed
    if (_exterior_strategy == ExteriorStrategy::CONTRACT && _external_degrees.empty()) {
        init_external_degrees();
    }

    const PEID size = mpi::get_comm_size(_p_graph.communicator());
    const PEID rank = mpi::get_comm_rank(_p_graph.communicator());

    std::vector<GhostSeedNode> initial_seed_nodes;
    for (const auto start_node: start_nodes) {
        initial_seed_nodes.emplace_back(0, start_node);
    }

    auto local_result = run_multi_seeded_bfs(rank, initial_seed_nodes);
}

std::vector<std::vector<BfsExtractor::GhostSeedNode>>
exchange_ghost_seed_nodes(const std::vector<std::vector<GhostSeedNode>>& all_ghost_seed_nodes) {
    const PEID size = mpi::get_comm_size(_p_graph.communicator());

    std::vector<std::vector<NodeID>> sendbufs(static_cast<std::size_t>(size));
    for (PEID pe = 0; pe < size; ++pe) {
        const auto& ghost_seed_nodes = all_ghost_seed_nodes[pe];
        if (!ghost_seed_nodes.empty()) {
            sendbufs.push_back(asserting_cast<NodeID>(pe));
            sendbufs.push_back(asserting_cast<NodeID>(ghost_seed_nodes.size()));
        }
        for (const auto& [distance, local_ghost_node]: ghost_seed_nodes) {
            const auto owner       = _p_graph.ghost_owner(local_ghost_node);
            const auto global_node = _p_graph.ghost_to_global(local_ghost_node);
            const auto remote_node = asserting_cast<NodeID>(global_node - _p_graph.node_distribution(owner));
            sendbufs[owner].push_back(distance);
            sendbufs[owner].push_back(remote_node);
        }
    }

    std::vector<std::vector<GhostSeedNode>> next_all_ghost_seed_nodes(static_cast<std::size_t>(size));
    mpi::sparse_alltoall<NodeID>(
        std::move(sendbufs),
        [&](const PEID pe, const auto& recvbuf) {
            for (std::size_t i = 0; i < recvbuf.size();) {
                const PEID        initial_pe     = asserting_cast<PEID>(recvbuf[i++]);
                const std::size_t num_seed_nodes = asserting_cast<std::size_t>(recvbuf[i++]);

                for (std::size_t j = 0; j < num_seed_nodes; ++j) {
                    const NodeID distance = recvbuf[i++];
                    const NodeID node     = recvbuf[i++];
                    next_all_ghost_seed_nodes[initial_pe].emplace_back(distance, node);
                }
            }
        },
        _p_graph.communicator()
    );

    return next_all_ghost_seed_nodes;
}

std::vector<BfsExtractor::GraphSegment> exchange_subgraphs(const std::vector<std::vector<NodeID>> &nodes) {
    const PEID size = mpi::get_comm_size(_p_graph.communicator());

    // Build subgraph segments for each PE
    std::vector<std::vector<NodeID>> sendbufs(static_cast<std::size_t>(size));
    for (PEID pe = 0; pe < size; ++pe) {
        
    }
}

BfsExtractedGraph BfsExtractor::extract_from_node(const NodeID start_node) {
    return {};
}

BfsExtractor::LocalBfsResult
BfsExtractor::run_multi_seeded_bfs(const PEID initial_pe, std::vector<BfsExtractor::GhostSeedNode>& ghost_seed_nodes) {
    // Catch case where there are no seed nodes
    if (ghost_seed_nodes.empty()) {
        return {};
    }

    // Maximum number of nodes to be explored by this BFS search
    const NodeID max_size = _p_graph.total_n() < _max_size ? _p_graph.total_n() : _max_size * ghost_seed_nodes.size();

    // Marker for nodes that were already explored by this BFS search
    auto& taken = _taken_ets.local();

    // Sort ghost seed nodes by distance
    std::sort(ghost_seed_nodes.begin(), ghost_seed_nodes.end(), [](const auto& lhs, const auto& rhs) {
        return std::get<0>(lhs) < std::get<0>(rhs);
    });

    // Initialize search from closest ghost seed nodes
    NodeID                     current_distance = std::get<0>(ghost_seed_nodes.front());
    std::stack<NodeID>         front;
    std::vector<NodeID>        visited_nodes;
    std::vector<GhostSeedNode> next_ghost_seed_nodes;

    auto add_seed_nodes_to_search_front = [&](const NodeID with_distance) {
        for (const auto& [distance, hops, node]: ghost_seed_nodes) {
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
        if (_p_graph.is_owned_node(node)) {
            explore_outgoing_edges(node, [&](const NodeID neighbor) {
                if (taken.get(node)) {
                    // Skip nodes already discovered
                    return true;
                }
                if (!_p_graph.is_owned_node(neighbor) && _p_graph.ghost_owner(neighbor) != initial_pe) {
                    // Record ghost seeds for the next hop
                    next_ghost_seed_nodes.emplace_back(current_distance, node);
                }
                if (current_distance + 1 < _max_radius) {
                    // Add to stack for the next search layer
                    front.push(neighbor);
                }

                // Record node as discovered
                visited_nodes.push_back(neighbor);
                taken.set(neighbor);

                return visited_nodes.size() < _max_size;
            });
        }

        if (front_size == 0) {
            front_size = front.size();
            ++current_distance;
            add_seed_nodes_to_search_front(current_distance + 1);
        }
    }

    LocalBfsResult ans;
    ans.explored_nodes  = std::move(visited_nodes);
    ans.explored_ghosts = std::move(next_ghost_seed_nodes);
    return ans;
}

template <typename Lambda>
void BfsExtractor::explore_outgoing_edges(const NodeID node, Lambda&& lambda) {
    const bool is_high_degree_node = _p_graph.degree(node) >= _high_degree_threshold;

    if (!is_high_degree_node || _high_degree_strategy == HighDegreeStrategy::TAKE_ALL) {
        for (const auto [e, v]: _p_graph.neighbors(node)) {
            if (!explore_neighbor(v)) {
                break;
            }
        }
    } else if (_high_degree_strategy == HighDegreeStrategy::CUT) {
        for (EdgeID e = _p_graph.first_edge(node); e < _p_graph.first_edge(node) + _high_degree_threshold; ++e) {
            if (!explore_neighbor(_p_graph.edge_target(e))) {
                break;
            }
        }
    } else if (_high_degree_strategy == HighDegreeStrategy::SAMPLE) {
        const double                        skip_prob = 1.0 * _high_degree_threshold / _p_graph.degree(node);
        std::geometric_distribution<EdgeID> skip_dist(skip_prob);

        for (EdgeID e = _p_graph.first_edge(node); e < _p_graph.first_invalid_edge(node); e += skip_dist(gen)) {
            if (!explore_neighbor(_p_graph.edge_target(e))) {
                break;
            }
        }
    } else {
        // do nothing for HighDegreeStrategy::IGNORE
    }
}

BfsExtractedGraph BfsExtractor::extract(const NodeID start_node) {
    return std::move(extract(std::vector<NodeID>{start_node}).front());
}

BfsExtractor::InducedSubgraph BfsExtractor::build_induced_subgraph(const std::vector<NodeID>& nodes) {
    // Build mappings from and to subgraph
    const auto&                        mapping_subgraph_to_graph = nodes;
    std::unordered_map<NodeID, NodeID> mapping_graph_to_subgraph;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        mapping_graph_to_subgraph[nodes[i]] = asserting_cast<NodeID>(i);
    }

    // Data strucutes for subgraph
    const NodeID            sub_n       = nodes.size() + _p_graph.k();
    const NodeID            sub_n_prime = nodes.size();
    std::vector<EdgeID>     sub_nodes(sub_n + 1);
    std::vector<NodeID>     sub_edges;
    std::vector<NodeWeight> sub_node_weights(sub_n);
    std::vector<EdgeWeight> sub_edge_weights;

    std::vector<EdgeWeight> edge_weights_to_block_nodes(_p_graph.k());

    for (const NodeID u: nodes) {
        sub_nodes[u]        = asserting_cast<EdgeID>(sub_edges.size());
        sub_node_weights[u] = _p_graph.node_weight(u);

        for (const auto [e, v]: _p_graph.neighbors(u)) {
            auto v_it = mapping_graph_to_subgraph.find(v);

            if (v_it != mapping_graph_to_subgraph.end()) {
                // v is also in the subgraph
                const auto [v_prime, sub_v] = *v_it;
                KASSERT(v_prime == v);
                sub_edges.push_back(sub_v);
                sub_edge_weights.push_back(_p_graph.edge_weight(e));
            } else {
                // v is not in the subgraph -> contributes to our edge to a block node
                edge_weights_to_block_nodes[_p_graph.block(v)] += _p_graph.edge_weight(e);
            }
        }

        for (const BlockID b: _p_graph.blocks()) {
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
        [&](const NodeID node) { return _p_graph.local_to_global_node(node); }
    );

    NoinitVector<BlockID> partition(sub_n);
    std::transform(
        mapping_subgraph_to_graph.begin(), mapping_subgraph_to_graph.end(), partition.begin(),
        [&](const NodeID node) { return _p_graph.block(node); }
    );

    return {std::move(subgraph), std::move(partition), std::move(mapping)};
}

void BfsExtractor::allow_overlap() {
    _overlap = true;
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
    _external_degrees.resize(_p_graph.n() * _p_graph.k());
    tbb::parallel_for<std::size_t>(0, _external_degrees.size(), [&](const std::size_t i) { _external_degrees[i] = 0; });

    _p_graph.pfor_nodes([&](const NodeID u) {
        for (const auto [e, v]: _p_graph.neighbors(u)) {
            const BlockID    v_block  = _p_graph.block(v);
            const EdgeWeight e_weight = _p_graph.edge_weight(e);
            external_degree(u, v_block) += e_weight;
        }
    });
}

EdgeWeight& BfsExtractor::external_degree(const NodeID u, const BlockID b) {
    KASSERT(u * _p_graph.k() + b < _external_degrees.size());
    return _external_degrees[u * _p_graph.k() + b];
}
} // namespace kaminpar::dist::graph

