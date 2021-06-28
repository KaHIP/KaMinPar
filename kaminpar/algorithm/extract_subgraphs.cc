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
#include "algorithm/extract_subgraphs.h"

#include "datastructure/graph.h"
#include "datastructure/static_array.h"
#include "definitions.h"
#include "parallel.h"
#include "utility/timer.h"

#include <mutex>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

namespace kaminpar::graph {

SequentialSubgraphExtractionResult extract_subgraphs_sequential(const PartitionedGraph &p_graph,
                                                                const SubgraphMemoryStartPosition memory_position,
                                                                SubgraphMemory &subgraph_memory,
                                                                TemporarySubgraphMemory &tmp_subgraph_memory) {
  ALWAYS_ASSERT(p_graph.k() == 2) << "Only suitable for bipartitions!";
  ALWAYS_ASSERT(tmp_subgraph_memory.in_use == false);
  tmp_subgraph_memory.in_use = true;

  const bool is_node_weighted = p_graph.graph().is_node_weighted();
  const bool is_edge_weighted = p_graph.graph().is_edge_weighted();

  const BlockID final_k = p_graph.final_k(0) + p_graph.final_k(1);
  tmp_subgraph_memory.ensure_size_nodes(p_graph.n() + final_k, is_node_weighted);

  auto &nodes = tmp_subgraph_memory.nodes;
  auto &edges = tmp_subgraph_memory.edges;
  auto &node_weights = tmp_subgraph_memory.node_weights;
  auto &edge_weights = tmp_subgraph_memory.edge_weights;
  auto &mapping = tmp_subgraph_memory.mapping;

  std::array<NodeID, 2> s_n{0, 0};
  std::array<EdgeID, 2> s_m{0, 0};

  // find graph sizes
  for (const NodeID u : p_graph.nodes()) {
    const BlockID b = p_graph.block(u);
    tmp_subgraph_memory.mapping[u] = s_n[b]++;

    for (const auto [e, v] : p_graph.neighbors(u)) {
      if (p_graph.block(v) == b) { ++s_m[b]; }
    }
  }

  // start position of subgraph[1] in common memory ds
  const NodeID n1 = s_n[0] + p_graph.final_k(0);
  const EdgeID m1 = s_m[0];

  nodes[0] = 0;
  nodes[n1] = 0;
  tmp_subgraph_memory.ensure_size_edges(s_m[0] + s_m[1], is_edge_weighted);

  // build extract graphs in temporary memory buffer
  std::array<EdgeID, 2> next_edge_id{0, 0};

  for (const NodeID u : p_graph.nodes()) {
    const BlockID b = p_graph.block(u);

    const NodeID n0 = b * n1; // either 0 or s_n[0] + final_k(0)
    const EdgeID m0 = b * m1; // either 0 or s_m[0]

    for (const auto [e, v] : p_graph.neighbors(u)) {
      if (p_graph.block(v) == b) {
        edges[m0 + next_edge_id[b]] = mapping[v];
        if (is_edge_weighted) { edge_weights[m0 + next_edge_id[b]] = p_graph.edge_weight(e); }
        ++next_edge_id[b];
      }
    }

    nodes[n0 + mapping[u] + 1] = next_edge_id[b];
    if (is_node_weighted) { node_weights[n0 + mapping[u]] = p_graph.node_weight(u); }
  }

  // copy graphs to subgraph_memory at memory_position
  // THIS OPERATION OVERWRITES p_graph!
  std::copy(nodes.begin(), nodes.begin() + p_graph.n() + final_k,
            subgraph_memory.nodes.begin() + memory_position.nodes_start_pos);
  std::copy(edges.begin(), edges.begin() + s_m[0] + s_m[1],
            subgraph_memory.edges.begin() + memory_position.edges_start_pos);
  if (is_node_weighted) {
    std::copy(node_weights.begin(), node_weights.begin() + p_graph.n() + final_k,
              subgraph_memory.node_weights.begin() + memory_position.nodes_start_pos);
  }
  if (is_edge_weighted) {
    std::copy(edge_weights.begin(), edge_weights.begin() + s_m[0] + s_m[1],
              subgraph_memory.edge_weights.begin() + memory_position.edges_start_pos);
  }

  tmp_subgraph_memory.in_use = false;

  std::array<SubgraphMemoryStartPosition, 2> subgraph_positions;
  subgraph_positions[0].nodes_start_pos = memory_position.nodes_start_pos;
  subgraph_positions[0].edges_start_pos = memory_position.edges_start_pos;
  subgraph_positions[1].nodes_start_pos = memory_position.nodes_start_pos + n1;
  subgraph_positions[1].edges_start_pos = memory_position.edges_start_pos + m1;

  auto create_graph = [&](const NodeID n0, const NodeID n, const EdgeID m0, const EdgeID m) {
    StaticArray<EdgeID> s_nodes(memory_position.nodes_start_pos + n0, n + 1, subgraph_memory.nodes);
    StaticArray<NodeID> s_edges(memory_position.edges_start_pos + m0, m, subgraph_memory.edges);
    StaticArray<NodeWeight> s_node_weights(is_node_weighted * (memory_position.nodes_start_pos + n0),
                                           is_node_weighted * n, subgraph_memory.node_weights);
    StaticArray<EdgeWeight> s_edge_weights(is_edge_weighted * (memory_position.edges_start_pos + m0),
                                           is_edge_weighted * m, subgraph_memory.edge_weights);
    return Graph{tag::seq, std::move(s_nodes), std::move(s_edges), std::move(s_node_weights),
                 std::move(s_edge_weights)};
  };

  std::array<Graph, 2> subgraphs{create_graph(0, s_n[0], 0, s_m[0]), create_graph(n1, s_n[1], m1, s_m[1])};

  return {std::move(subgraphs), std::move(subgraph_positions)};
}

/*
 * Builds a block-induced subgraph for each block of a partitioned graph. Return type contains a mapping that maps
 * nodes from p_graph to nodes in the respective subgraph; we need this because the order in which nodes in subgraphs
 * appear is non-deterministic due to parallelization.
 */
SubgraphExtractionResult extract_subgraphs(const PartitionedGraph &p_graph, SubgraphMemory &subgraph_memory) {
  const Graph &graph = p_graph.graph();

  const auto t_allocation = SIMPLE_TIMER_START();
  START_TIMER(TIMER_ALLOCATION);
  scalable_vector<NodeID> mapping(p_graph.n());
  scalable_vector<SubgraphMemoryStartPosition> start_positions(p_graph.k() + 1);

  using AtomicNodeCounter = parallel::IntegralAtomicWrapper<NodeID>;
  std::vector<AtomicNodeCounter, tbb::cache_aligned_allocator<AtomicNodeCounter>> bucket_index(p_graph.n());

  scalable_vector<Graph> subgraphs(p_graph.k());
  STOP_TIMER();
  SIMPLE_TIMER_STOP("Subgraph extraction allocation", t_allocation);

  // count number of nodes and edges in each block
  START_TIMER("Count block size");
  tbb::enumerable_thread_specific<scalable_vector<NodeID>> tl_num_nodes_in_block{
      [&] { return scalable_vector<NodeID>(p_graph.k()); }};
  tbb::enumerable_thread_specific<scalable_vector<EdgeID>> tl_num_edges_in_block{
      [&] { return scalable_vector<EdgeID>(p_graph.k()); }};

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, graph.n()), [&](auto &r) {
    auto &num_nodes_in_block = tl_num_nodes_in_block.local();
    auto &num_edges_in_block = tl_num_edges_in_block.local();

    for (NodeID u = r.begin(); u != r.end(); ++u) {
      const BlockID u_block = p_graph.block(u);
      ++num_nodes_in_block[u_block];
      for (const NodeID v : graph.adjacent_nodes(u)) {
        if (p_graph.block(v) == u_block) { ++num_edges_in_block[u_block]; }
      }
    }
  });
  STOP_TIMER();

  START_TIMER("Merge block sizes");
  tbb::parallel_for(static_cast<BlockID>(0), p_graph.k(), [&](const BlockID b) {
    NodeID num_nodes = p_graph.final_k(b); // padding for sequential subgraph extraction
    EdgeID num_edges = 0;
    for (auto &local_num_nodes : tl_num_nodes_in_block) { num_nodes += local_num_nodes[b]; }
    for (auto &local_num_edges : tl_num_edges_in_block) { num_edges += local_num_edges[b]; }
    start_positions[b + 1].nodes_start_pos = num_nodes;
    start_positions[b + 1].edges_start_pos = num_edges;
  });
  parallel::prefix_sum(start_positions.begin(), start_positions.end(), start_positions.begin());
  STOP_TIMER();

  // build temporary bucket array in nodes array
  START_TIMER("Build bucket array");
  tbb::parallel_for(static_cast<NodeID>(0), p_graph.n(), [&](const NodeID u) {
    const BlockID b = p_graph.block(u);
    const NodeID pos_in_subgraph = bucket_index[b]++;
    const NodeID pos = start_positions[b].nodes_start_pos + pos_in_subgraph;
    subgraph_memory.nodes[pos] = u;
    mapping[u] = pos_in_subgraph; // concurrent random access write
  });
  STOP_TIMER();

  const bool is_node_weighted = p_graph.graph().is_node_weighted();
  const bool is_edge_weighted = p_graph.graph().is_edge_weighted();

  // build graph
  START_TIMER("Construct subgraphs");
  tbb::parallel_for(static_cast<BlockID>(0), p_graph.k(), [&](const BlockID b) {
    const NodeID nodes_start_pos = start_positions[b].nodes_start_pos;
    EdgeID e = 0;                                  // edge = in subgraph
    for (NodeID u = 0; u < bucket_index[b]; ++u) { // u = in subgraph
      const NodeID pos = nodes_start_pos + u;
      const NodeID u_prime = subgraph_memory.nodes[pos]; // u_prime = in graph
      subgraph_memory.nodes[pos] = e;
      if (is_node_weighted) { subgraph_memory.node_weights[pos] = graph.node_weight(u_prime); }

      const EdgeID e0 = start_positions[b].edges_start_pos;

      for (const auto [e_prime, v_prime] : graph.neighbors(u_prime)) { // e_prime, v_prime = in graph
        if (p_graph.block(v_prime) == b) {                             // only keep internal edges
          if (is_edge_weighted) { subgraph_memory.edge_weights[e0 + e] = graph.edge_weight(e_prime); }
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
    const NodeID n = start_positions[b + 1].nodes_start_pos - n0 - p_graph.final_k(b);
    const EdgeID m = start_positions[b + 1].edges_start_pos - m0;

    StaticArray<EdgeID> nodes(n0, n + 1, subgraph_memory.nodes);
    StaticArray<NodeID> edges(m0, m, subgraph_memory.edges);
    StaticArray<NodeWeight> node_weights(is_node_weighted * n0, is_node_weighted * n, subgraph_memory.node_weights);
    StaticArray<EdgeWeight> edge_weights(is_edge_weighted * m0, is_edge_weighted * m, subgraph_memory.edge_weights);
    subgraphs[b] = Graph{std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights)};
  });
  STOP_TIMER();

  HEAVY_ASSERT([&] {
    for (const BlockID b : p_graph.blocks()) {
      LOG << "Validate " << b;
      ALWAYS_ASSERT(validate_graph(subgraphs[b]));
    }
    return true;
  });

  return {std::move(subgraphs), std::move(mapping), std::move(start_positions)};
}
} // namespace kaminpar::graph