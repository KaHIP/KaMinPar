/*******************************************************************************
 * Multi-way flow refiner.
 *
 * @file:   multiway_flow_refiner.cc
 * @author: Daniel Salwasser
 * @date:   10.04.2025
 ******************************************************************************/
#include "kaminpar-shm/refinement/flow/multiway_flow_refiner.h"

#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/refinement/balancer/sequential_greedy_balancer.h"
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

  auto [initial_cut_value, border_regions] = compute_border_regions(p_ctx);
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

  const std::unordered_set<EdgeID> cut_edges = TIMED_SCOPE("Compute Multi-Way Cut") {
    return _multiway_cut_algorithm->compute(local_p_graph, flow_network.graph, terminal_sets);
  };

  std::unordered_map<NodeID, BlockID> local_to_block_mapping;
  std::unordered_set<NodeID> nodes;
  for (BlockID block = 0; block < _p_graph->k(); ++block) {
    Cut cut = compute_cut_nodes(flow_network.graph, terminal_sets[block], cut_edges);
    DBG << "Block " << block << " has weight " << cut.weight
        << "; max: " << p_ctx.max_block_weight(block);

    for (const NodeID u : cut.nodes) {
      local_to_block_mapping.emplace(u, block);
    }
  }

  for (const auto &[u, u_local] : flow_network.global_to_local_mapping) {
    const BlockID new_block = local_to_block_mapping[u_local];
    p_graph.set_block(u, new_block);
  }

  if (_f_ctx.rebalance) {
    SequentialGreedyBalancerImpl<PartitionedGraph, CSRGraph> balancer;
    const auto [rebalanced, gain] = balancer.balance(p_graph, graph, p_ctx.max_block_weights());

    DBG << "Rebalanced partition with gain " << gain;
  }

  return false;
}

MultiwayFlowRefiner::BorderRegions
MultiwayFlowRefiner::compute_border_regions(const PartitionContext &p_ctx) const {
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

  EdgeWeight cut_value = 0;
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
    _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      const BlockID v_block = _p_graph->block(v);
      if (u_block == v_block) {
        return;
      }

      BorderRegion &v_border_region = border_regions[v_block];
      if (v_border_region.contains(v)) {
        is_border_region_node = true;
        cut_value += w;
        return;
      }

      const NodeWeight v_weight = _graph->node_weight(v);
      if (v_border_region.fits(v_weight)) {
        v_border_region.insert(v, v_weight);
        is_border_region_node = true;
        cut_value += w;
      }
    });

    if (is_border_region_node) {
      u_border_region.insert(u, u_weight);
    }
  }

  return BorderRegions(cut_value, std::move(border_regions));
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

  cur_node = _p_graph->k();
  for (BlockID block = 0; block < _p_graph->k(); ++block) {
    NodeWeight border_region_weight = 0;
    EdgeID num_terminal_edges = 0;

    for (const NodeID u : border_regions[block].nodes()) {
      EdgeID num_neighbors = 0;
      _graph->adjacent_nodes(u, [&](const NodeID v) {
        num_neighbors += global_to_local_mapping.contains(v) ? 1 : 0;
      });

      const bool has_non_border_region_neighbor = num_neighbors != _graph->degree(u);
      if (has_non_border_region_neighbor) { // Node has an edge to its corresponding terminal
        num_neighbors += 1;
        num_terminal_edges += 1;
      }

      const NodeWeight u_weight = _graph->node_weight(u);
      nodes[cur_node + 1] = num_neighbors;
      node_weights[cur_node] = u_weight;

      border_region_weight += u_weight;
      cur_node += 1;
    }

    KASSERT(num_terminal_edges > static_cast<EdgeID>(0), assert::light);

    nodes[block + 1] = num_terminal_edges;
    node_weights[block] = _p_graph->block_weight(block) - border_region_weight;
  }
  KASSERT(cur_node == num_nodes);

  nodes[0] = 0;
  std::partial_sum(nodes.begin(), nodes.end(), nodes.begin());

  const EdgeID num_edges = nodes.back();
  StaticArray<NodeID> edges(num_edges, static_array::noinit);
  StaticArray<EdgeWeight> edge_weights(num_edges, static_array::noinit);

  cur_node = _p_graph->k();
  EdgeID cur_edge = nodes[_p_graph->k()];
  EdgeID cur_source_edge = 0;
  for (BlockID block = 0; block < _p_graph->k(); ++block) {
    for (const NodeID u : border_regions[block].nodes()) {
      EdgeWeight terminal_edge_weight = 0;
      _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        if (auto it = global_to_local_mapping.find(v); it != global_to_local_mapping.end()) {
          const NodeID v_local = it->second;

          edges[cur_edge] = v_local;
          edge_weights[cur_edge] = w;
          cur_edge += 1;
        } else {
          terminal_edge_weight += w;
        }
      });

      const NodeID u_degree = cur_edge - nodes[cur_node];
      const bool has_non_border_region_neighbor = u_degree != _graph->degree(u);
      if (has_non_border_region_neighbor) { // Connect node to its corresponding terminal
        edges[cur_edge] = block;
        edge_weights[cur_edge] = terminal_edge_weight;
        cur_edge += 1;

        edges[cur_source_edge] = cur_node;
        edge_weights[cur_source_edge] = terminal_edge_weight;
        cur_source_edge += 1;
      }

      cur_node += 1;
    }
  }
  KASSERT(cur_node == num_nodes);
  KASSERT(cur_edge == num_edges);
  KASSERT(cur_source_edge == nodes[_p_graph->k()]);

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
