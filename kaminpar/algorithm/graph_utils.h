/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
/** @file */
#pragma once

#include "context.h"
#include "datastructure/graph.h"
#include "datastructure/marker.h"
#include "datastructure/queue.h"
#include "utility/random.h"

#include <utility>
#include <vector>

namespace kaminpar {
bool validate_graph(const Graph &graph);

void copy_subgraph_partitions(PartitionedGraph &p_graph,
                              const scalable_vector<StaticArray<BlockID>> &p_subgraph_partitions,
                              const BlockID k_per_subgraph, const BlockID final_k_per_subgraph,
                              const scalable_vector<NodeID> &mapping);

struct SubgraphMemoryStartPosition {
  std::size_t nodes_start_pos{0};
  std::size_t edges_start_pos{0};

  // operator overloads for parallel::prefix_sum()
  SubgraphMemoryStartPosition operator+(const SubgraphMemoryStartPosition &other) {
    return {nodes_start_pos + other.nodes_start_pos, edges_start_pos + other.edges_start_pos};
  }

  SubgraphMemoryStartPosition &operator+=(const SubgraphMemoryStartPosition &other) {
    nodes_start_pos += other.nodes_start_pos;
    edges_start_pos += other.edges_start_pos;
    return *this;
  }
};

struct SubgraphMemory {
  SubgraphMemory(const NodeID n, const BlockID k, const EdgeID m, const bool is_node_weighted = true,
                 const bool is_edge_weighted = true)
      : nodes(n + k),
        edges(m),
        node_weights(is_node_weighted * (n + k)),
        edge_weights(is_edge_weighted * m) {}

  SubgraphMemory(const PartitionedGraph &p_graph)
      : SubgraphMemory(p_graph.n(), p_graph.k(), p_graph.m(), p_graph.graph().is_node_weighted(),
                       p_graph.graph().is_edge_weighted()) {}

  StaticArray<EdgeID> nodes;
  StaticArray<NodeID> edges;
  StaticArray<NodeWeight> node_weights;
  StaticArray<EdgeWeight> edge_weights;
};

struct SubgraphExtractionResult {
  scalable_vector<Graph> subgraphs;
  scalable_vector<NodeID> node_mapping;
  scalable_vector<SubgraphMemoryStartPosition> positions;
};

struct SequentialSubgraphExtractionResult {
  std::array<Graph, 2> subgraphs;
  std::array<SubgraphMemoryStartPosition, 2> positions;
};

struct TemporarySubgraphMemory {
  constexpr static double kOverallocationFactor = 1.05;

  void ensure_size_nodes(const NodeID n, const bool is_node_weighed) {
    if (nodes.size() < n + 1) {
      nodes.resize(n * kOverallocationFactor + 1);
      ++num_node_reallocs;
    }
    if (is_node_weighed && node_weights.size() < n) { node_weights.resize(n * kOverallocationFactor); }
    if (mapping.size() < n) { mapping.resize(n * kOverallocationFactor); }
  }

  void ensure_size_edges(const EdgeID m, const bool is_edge_weighted) {
    if (edges.size() < m) {
      edges.resize(m * kOverallocationFactor);
      ++num_edge_reallocs;
    }
    if (is_edge_weighted && edge_weights.size() < m) { edge_weights.resize(m * kOverallocationFactor); }
  }

  std::vector<EdgeID> nodes;
  std::vector<NodeID> edges;
  std::vector<NodeWeight> node_weights;
  std::vector<EdgeWeight> edge_weights;
  std::vector<NodeID> mapping;

  bool in_use{false};

  std::size_t num_node_reallocs = 0;
  std::size_t num_edge_reallocs = 0;

  std::size_t memory_in_kb() const {
    return nodes.size() * sizeof(EdgeID) / 1000 +            //
           edges.size() * sizeof(NodeID) / 1000 +            //
           node_weights.size() * sizeof(NodeWeight) / 1000 + //
           edge_weights.size() * sizeof(EdgeWeight) / 1000 + //
           mapping.size() * sizeof(NodeID) / 1000;           //
  }
};

SubgraphExtractionResult extract_subgraphs(const PartitionedGraph &p_graph, SubgraphMemory &subgraph_memory);

SequentialSubgraphExtractionResult extract_subgraphs_sequential(const PartitionedGraph &p_graph,
                                                                SubgraphMemoryStartPosition memory_position,
                                                                SubgraphMemory &subgraph_memory,
                                                                TemporarySubgraphMemory &tmp_subgraph_memory);

using NodePermutation = StaticArray<NodeID>;

struct NodePermutations {
  NodePermutation old_to_new;
  NodePermutation new_to_old;
};

NodePermutations sort_by_degree_buckets(const StaticArray<EdgeID> &nodes, const bool deg0_position = false);

void build_permuted_graph(const StaticArray<EdgeID> &old_nodes, const StaticArray<NodeID> &old_edges,
                          const StaticArray<NodeWeight> &old_node_weights,
                          const StaticArray<EdgeWeight> &old_edge_weights, const NodePermutations &permutation,
                          StaticArray<EdgeID> &new_nodes, StaticArray<NodeID> &new_edges,
                          StaticArray<NodeWeight> &new_node_weights, StaticArray<EdgeWeight> &new_edge_weights);

std::pair<NodeID, NodeWeight> find_isolated_nodes_info(const StaticArray<EdgeID> &nodes,
                                                       const StaticArray<NodeWeight> &node_weights);

std::pair<NodeID, NodeID> find_furthest_away_node(const Graph &graph, NodeID start_node, Queue<NodeID> &queue,
                                                  Marker<> &marker);

NodePermutations rearrange_and_remove_isolated_nodes(const bool remove_isolated_nodes, PartitionContext &p_ctx,
                                                     StaticArray<EdgeID> &nodes, StaticArray<NodeID> &edges,
                                                     StaticArray<NodeWeight> &node_weights,
                                                     StaticArray<EdgeWeight> &edge_weights,
                                                     NodeWeight total_node_weight = -1);

PartitionedGraph revert_isolated_nodes_removal(PartitionedGraph p_graph, const NodeID num_isolated_nodes,
                                               const PartitionContext &p_ctx);

/*!
 * Fast heuristic for finding two nodes with large distance: selects a random node (if seed_node is not specified),
 * performs a BFS and selects the last node processed as pseudo peripheral node. If the graph is disconnected, we select
 * a node in another connected component.
 *
 * @tparam seed_node If specified, start from this node instead of a random one (for unit tests).
 * @param graph
 * @param num_iterations Repeat the algorithm this many times for a chance of finding a pair of nodes with even larger
 * distance.
 * @return Pair of nodes with large distance between them.
 */
template<NodeID seed_node = kInvalidNodeID> // default: pick random nodes
std::pair<NodeID, NodeID> find_far_away_nodes(const Graph &graph, const std::size_t num_iterations = 1) {
  Queue<NodeID> queue(graph.n());
  Marker<> marker(graph.n());

  if constexpr (seed_node != kInvalidNodeID) { // for unit test
    return {seed_node, find_furthest_away_node(graph, seed_node, queue, marker).first};
  }

  NodeID best_distance = 0;
  std::pair<NodeID, NodeID> best_pair{0, 0};
  for (std::size_t i = 0; i < num_iterations; ++i) {
    const NodeID u = Randomize::instance().random_node(graph);
    const auto [v, distance] = find_furthest_away_node(graph, u, queue, marker);

    if (distance > best_distance || (distance == best_distance && Randomize::instance().random_bool())) {
      best_distance = distance;
      best_pair = {u, v};
    }
  }

  return best_pair;
}
} // namespace kaminpar
