/*******************************************************************************
 * @file:   bfs_extractor.cc
 * @author: Daniel Seemaier
 * @date:   23.08.2022
 * @brief:  Algorithm to extract a region of the graph using a BFS search.
 ******************************************************************************/
#include "dkaminpar/graphutils/bfs_extractor.h"

#include <random>
#include <stack>

#include <kassert/kassert.hpp>
#include <tbb/parallel_for.h>

#include "dkaminpar/datastructure/distributed_graph.h"

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

    std::vector<BfsExtractedGraph> ans(start_nodes.size());

    tbb::parallel_for<std::size_t>(0, start_nodes.size(), [&](const std::size_t i) {
        const NodeID start_node = start_nodes[i];
        grow_multi_seeded_region({{start_node}}, 0);
    });

    return ans;
}

BfsExtractedGraph BfsExtractor::extract_from_node(const NodeID start_node) {
    return {};
}

void BfsExtractor::grow_multi_seeded_region(const std::vector<std::vector<NodeID>>& seed_nodes_per_distance) {
    // Find closest seed nodes
    NodeID distance;
    for (distance = 0; distance < seed_nodes_per_distance.size() && seed_nodes_per_distance[distance].empty();
         ++distance) {
    }

    std::stack<NodeID> search;

    auto& gen   = Random::instance().generator();
    auto& taken = _taken_ets.local();
    taken.reset();

    // Result vectors
    std::vector<NodeID>                    ans;
    std::vector<std::pair<NodeID, NodeID>> ans_hop_seeds;

    auto add_layer_seeds_to_stack = [&](const NodeID layer) {
        if (layer < seed_nodes_per_distance.size()) {
            for (const NodeID node: seed_nodes_per_distance[layer]) {
                if (!taken.get(node)) {
                    search.push(node);
                    taken.set(node);
                }
            }
        }
    };

    add_layer_seeds_to_stack(distance);
    NodeID front_size = search.size();
    add_layer_seeds_to_stack(distance + 1);

    while (distance < _max_radius && !search.empty()) {
        const NodeID node = search.top();
        search.pop();
        --front_size;

        ans.push_back(node);
        if (ans.size() == _max_size) {
            break;
        }

        // Grow search to neighbors of `node`
        if (_p_graph.is_owned_node(node)) {
            auto explore_neighbor = [&](const NodeID v) {
                // Only consider nodes once
                if (taken.get(v)) {
                    return;
                }

                taken.set(v);
                search.push(v);
            };

            // Select outgoing edges along which we grow search front
            const bool is_high_degree_node = _p_graph.degree(node) >= _high_degree_threshold;
            if (!is_high_degree_node || _high_degree_strategy == HighDegreeStrategy::TAKE_ALL) {
                for (const auto [e, v]: _p_graph.neighbors(node)) {
                    explore_neighbor(v);
                }
            } else if (_high_degree_strategy == HighDegreeStrategy::CUT) {
                for (EdgeID e = _p_graph.first_edge(node); e < _p_graph.first_edge(node) + _high_degree_threshold;
                     ++e) {
                    explore_neighbor(_p_graph.edge_target(e));
                }
            } else if (_high_degree_strategy == HighDegreeStrategy::SAMPLE) {
                const double                        skip_prob = 1.0 * _high_degree_threshold / _p_graph.degree(node);
                std::geometric_distribution<EdgeID> skip_dist(skip_prob);

                for (EdgeID e = _p_graph.first_edge(node); e < _p_graph.first_invalid_edge(node); e += skip_dist(gen)) {
                    explore_neighbor(_p_graph.edge_target(e));
                }
            } else {
                // do nothing for HighDegreeStrategy::IGNORE
            }
        } else {
            // New seed node
            ans_hop_seeds.emplace_back(distance, node);
        }

        // Increase distance to seed if current layer was fully consumed
        if (front_size == 0) {
            ++distance;
            add_layer_seeds_to_stack(distance + 1);
        }
    }
}

BfsExtractedGraph BfsExtractor::extract(const NodeID start_node) {
    return std::move(extract(std::vector<NodeID>{start_node}).front());
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

