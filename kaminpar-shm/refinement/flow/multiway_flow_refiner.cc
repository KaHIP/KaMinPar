/*******************************************************************************
 * Multi-way flow refiner.
 *
 * @file:   multiway_flow_refiner.cc
 * @author: Daniel Salwasser
 * @date:   10.04.2025
 ******************************************************************************/
#include "kaminpar-shm/refinement/flow/multiway_flow_refiner.h"

#include <algorithm>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/flow/multiway_cut/isolating_cut_heuristic.h"
#include "kaminpar-shm/refinement/flow/multiway_cut/labelling_function_heuristic.h"
#include "kaminpar-shm/refinement/flow/util/border_region.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

MultiwayFlowRefiner::MultiwayFlowRefiner(const Context &ctx)
    : _f_ctx(ctx.refinement.multiway_flow) {
  switch (_f_ctx.cut_algorithm) {
  case CutAlgorithm::ISOLATING_CUT_HEURISTIC:
    _multiway_cut_algorithm =
        std::make_unique<IsolatingCutHeuristic>(_f_ctx.isolating_cut_heuristic);
    break;
  case CutAlgorithm::LABELLING_FUNCTION_HEURISTIC:
    _multiway_cut_algorithm =
        std::make_unique<LabellingFunctionHeuristic>(_f_ctx.labelling_function_heuristic);
    break;
  }
}

MultiwayFlowRefiner::~MultiwayFlowRefiner() = default;

std::string MultiwayFlowRefiner::name() const {
  return "Multi-Way Flow Refinement";
}

void MultiwayFlowRefiner::initialize([[maybe_unused]] const PartitionedGraph &p_graph) {}

bool MultiwayFlowRefiner::refine(
    PartitionedGraph &p_graph, [[maybe_unused]] const PartitionContext &p_ctx
) {
  return reified(
      p_graph,
      [&](const auto &csr_graph) { return refine(p_graph, csr_graph, p_ctx); },
      [&]([[maybe_unused]] const auto &compressed_graph) {
        LOG_WARNING << "Cannot refine a compressed graph using the multiway flow refiner.";
        return false;
      }
  );
}

bool MultiwayFlowRefiner::refine(
    PartitionedGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx
) {
  SCOPED_TIMER("Multi-Way Flow Refinement");
  SCOPED_HEAP_PROFILER("Multi-Way Flow Refinement");

  _p_graph = &p_graph;
  _graph = &graph;

  const EdgeWeight initial_cut_value = metrics::edge_cut(p_graph);

  std::vector<BorderRegion> border_regions = compute_border_regions(p_ctx);
  for (BorderRegion &border_region : border_regions) {
    expand_border_region(border_region);
  }

  FlowNetwork flow_network = construct_flow_network(border_regions);

  std::vector<std::unordered_set<NodeID>> terminal_sets;
  for (BlockID block = 0; block < _p_graph->k(); ++block) {
    terminal_sets.push_back({block});
  }

  PartitionedCSRGraph local_p_graph(flow_network.graph, _p_graph->k());
  for (const auto &[u, u_local] : flow_network.global_to_local_mapping) {
    local_p_graph.set_block(u_local, p_graph.block(u));
  }

  const auto [cut_value, cut_edges] = TIMED_SCOPE("Compute Multi-Way Cut") {
    return _multiway_cut_algorithm->compute(local_p_graph, flow_network.graph, terminal_sets);
  };

  if (cut_edges.empty()) {
    DBG << "Failed to compute a multi-way cut";
    return false;
  }

  const EdgeWeight gain = initial_cut_value - cut_value;
  DBG << "Found an cut with value gain " << gain << " (" << initial_cut_value << " -> " << cut_value
      << ")";
  if (gain <= 0) {
    return false;
  }

  std::unordered_map<NodeID, BlockID> local_to_block_mapping;
  std::unordered_set<NodeID> nodes;
  for (BlockID block = 0; block < _p_graph->k(); ++block) {
    Cut cut = compute_cut_nodes(flow_network.graph, terminal_sets[block], cut_edges);
    DBG << "Block " << block << " has weight " << cut.weight << "/"
        << p_ctx.max_block_weight(block);

    for (const NodeID u : cut.nodes) {
      KASSERT(!local_to_block_mapping.contains(u));
      local_to_block_mapping.emplace(u, block);
    }
  }

  for (const auto &[u, u_local] : flow_network.global_to_local_mapping) {
    const BlockID new_block = local_to_block_mapping[u_local];
    p_graph.set_block(u, new_block);
  }

  KASSERT(
      cut_value == metrics::edge_cut(p_graph),
      "cut value does not equal the actual cut value",
      assert::heavy
  );

  return true;
}

std::vector<BorderRegion> MultiwayFlowRefiner::compute_border_regions(const PartitionContext &p_ctx
) const {
  SCOPED_TIMER("Compute border regions");

  std::vector<BorderRegion> border_regions;
  border_regions.reserve(_p_graph->k());

  for (BlockID block = 0; block < _p_graph->k(); ++block) {
    const BlockWeight max_border_region_weight =
        (1 + _f_ctx.border_region_scaling_factor * p_ctx.epsilon()) *
            p_ctx.perfectly_balanced_block_weight(block) -
        (_graph->total_node_weight() - _p_graph->block_weight(block));
    border_regions.emplace_back(block, max_border_region_weight);
  }

  for (const NodeID u : _graph->nodes()) {
    const BlockID u_block = _p_graph->block(u);

    BorderRegion &u_border_region = border_regions[u_block];
    if (u_border_region.contains(u)) {
      continue;
    }

    const NodeWeight u_weight = _graph->node_weight(u);
    if (!u_border_region.fits(u_weight)) {
      continue;
    }

    bool is_border_region_node = false;
    _graph->adjacent_nodes(u, [&](const NodeID v) {
      const BlockID v_block = _p_graph->block(v);
      if (u_block == v_block) {
        return;
      }

      BorderRegion &v_border_region = border_regions[v_block];
      if (v_border_region.contains(v)) {
        is_border_region_node = true;
        return;
      }

      const NodeWeight v_weight = _graph->node_weight(v);
      if (v_border_region.fits(v_weight)) {
        v_border_region.insert(v, v_weight);
        is_border_region_node = true;
      }
    });

    if (is_border_region_node) {
      u_border_region.insert(u, u_weight);
    }
  }

  return border_regions;
}

void MultiwayFlowRefiner::expand_border_region(BorderRegion &border_region) const {
  SCOPED_TIMER("Expand border region");

  std::queue<std::pair<NodeID, NodeID>> bfs_queue;
  for (const NodeID u : border_region.nodes()) {
    bfs_queue.emplace(u, 0);
  }

  const BlockID block = border_region.block();
  while (!bfs_queue.empty()) {
    const auto [u, u_distance] = bfs_queue.front();
    bfs_queue.pop();

    _graph->adjacent_nodes(u, [&](const NodeID v) {
      if (_p_graph->block(v) != block || border_region.contains(v)) {
        return;
      }

      const NodeWeight v_weight = _graph->node_weight(v);
      if (border_region.fits(v_weight)) {
        border_region.insert(v, v_weight);

        if (u_distance < _f_ctx.max_border_distance) {
          bfs_queue.emplace(v, u_distance + 1);
        }
      }
    });
  }
}

MultiwayFlowRefiner::FlowNetwork
MultiwayFlowRefiner::construct_flow_network(const std::vector<BorderRegion> &border_regions) {
  SCOPED_TIMER("Construct Flow Network");

  NodeID cur_node = _p_graph->k();

  std::unordered_map<NodeID, NodeID> global_to_local_mapping;
  for (const BorderRegion &border_region : border_regions) {
    for (const NodeID u : border_region.nodes()) {
      global_to_local_mapping.emplace(u, cur_node++);
    }
  }

  const NodeID num_nodes = cur_node;
  StaticArray<EdgeID> nodes(num_nodes + 1, static_array::noinit);
  StaticArray<NodeWeight> node_weights(num_nodes, static_array::noinit);

  std::vector<bool> adjacency(_p_graph->k());
  std::vector<EdgeID> num_terminal_edges(_p_graph->k(), 0);

  cur_node = _p_graph->k();
  for (BlockID block = 0; block < _p_graph->k(); ++block) {
    for (const NodeID u : border_regions[block].nodes()) {
      std::fill(adjacency.begin(), adjacency.end(), false);

      EdgeID num_neighbors = 0;
      _graph->adjacent_nodes(u, [&](const NodeID v) {
        if (global_to_local_mapping.contains(v)) {
          num_neighbors += 1;
          return;
        }

        const BlockID v_block = _p_graph->block(v);
        adjacency[v_block] = true;
      });

      for (BlockID adjacent_block = 0; adjacent_block < _p_graph->k(); ++adjacent_block) {
        if (adjacency[adjacent_block]) {
          num_neighbors += 1;
          num_terminal_edges[adjacent_block] += 1;
        }
      }

      const NodeWeight u_weight = _graph->node_weight(u);
      nodes[cur_node + 1] = num_neighbors;
      node_weights[cur_node] = u_weight;

      cur_node += 1;
    }
  }
  KASSERT(cur_node == num_nodes);

  for (BlockID block = 0; block < _p_graph->k(); ++block) {
    nodes[block + 1] = num_terminal_edges[block];
    node_weights[block] = _p_graph->block_weight(block) - border_regions[block].weight();
  }

  nodes[0] = 0;
  std::partial_sum(nodes.begin(), nodes.end(), nodes.begin());

  const EdgeID num_edges = nodes.back();
  StaticArray<NodeID> edges(num_edges, static_array::noinit);
  StaticArray<EdgeWeight> edge_weights(num_edges, static_array::noinit);

  std::vector<EdgeID> cur_terminal_edge(_p_graph->k());
  for (BlockID block = 0; block < _p_graph->k(); ++block) {
    cur_terminal_edge[block] = nodes[block];
  }
  std::vector<EdgeWeight> terminal_edge_weight(_p_graph->k(), 0);

  cur_node = _p_graph->k();
  EdgeID cur_edge = nodes[_p_graph->k()];
  for (BlockID block = 0; block < _p_graph->k(); ++block) {
    for (const NodeID u : border_regions[block].nodes()) {
      std::fill(adjacency.begin(), adjacency.end(), false);
      std::fill(terminal_edge_weight.begin(), terminal_edge_weight.end(), 0);

      _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        if (auto it = global_to_local_mapping.find(v); it != global_to_local_mapping.end()) {
          const NodeID v_local = it->second;

          edges[cur_edge] = v_local;
          edge_weights[cur_edge] = w;

          cur_edge += 1;
          return;
        }

        const BlockID v_block = _p_graph->block(v);
        adjacency[v_block] = true;
        terminal_edge_weight[v_block] += w;
      });

      for (BlockID adjacent_block = 0; adjacent_block < _p_graph->k(); ++adjacent_block) {
        if (adjacency[adjacent_block]) {
          edges[cur_edge] = block;
          edge_weights[cur_edge] = terminal_edge_weight[adjacent_block];
          cur_edge += 1;

          edges[cur_terminal_edge[adjacent_block]] = cur_node;
          edge_weights[cur_terminal_edge[adjacent_block]] = terminal_edge_weight[adjacent_block];
          cur_terminal_edge[adjacent_block] += 1;
        }
      }

      cur_node += 1;
    }
  }
  KASSERT(cur_node == num_nodes);
  KASSERT(cur_edge == num_edges);

  CSRGraph graph(
      CSRGraph::seq(),
      std::move(nodes),
      std::move(edges),
      std::move(node_weights),
      std::move(edge_weights)
  );
  KASSERT(debug::validate_graph(graph), "constructed invalid flow network", assert::heavy);

  return FlowNetwork(std::move(graph), std::move(global_to_local_mapping));
}

MultiwayFlowRefiner::Cut MultiwayFlowRefiner::compute_cut_nodes(
    const CSRGraph &graph,
    const std::unordered_set<NodeID> &terminals,
    const std::unordered_set<EdgeID> &cut_edges
) {
  NodeWeight cut_weight = 0;
  std::unordered_set<NodeID> cut_nodes;

  std::queue<NodeID> bfs_queue;
  for (const NodeID terminal : terminals) {
    cut_weight += graph.node_weight(terminal);
    cut_nodes.insert(terminal);
    bfs_queue.push(terminal);
  }

  while (!bfs_queue.empty()) {
    const NodeID u = bfs_queue.front();
    bfs_queue.pop();

    graph.neighbors(u, [&](const EdgeID e, const NodeID v) {
      if (cut_nodes.contains(v) || cut_edges.contains(e)) {
        return;
      }

      cut_weight += graph.node_weight(v);
      cut_nodes.insert(v);
      bfs_queue.push(v);
    });
  }

  return Cut(cut_weight, std::move(cut_nodes));
}

} // namespace kaminpar::shm
