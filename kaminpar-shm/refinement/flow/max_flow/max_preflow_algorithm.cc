#include "kaminpar-shm/refinement/flow/max_flow/max_preflow_algorithm.h"

#include <cstddef>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::shm::debug {

bool is_valid_labeling(
    const CSRGraph &graph,
    const NodeStatus &node_status,
    std::span<const EdgeWeight> flow,
    std::span<const NodeID> labeling
) {
  if (graph.n() > labeling.size()) {
    LOG_WARNING << "Labeling size does not equal the number of nodes";
    return false;
  }

  bool is_valid = true;

  const NodeID n = graph.n();
  for (const NodeID source : node_status.source_nodes()) {
    if (labeling[source] != n) {
      LOG_WARNING << "Source condition violated for node " << source;
      is_valid = false;
    }
  }

  std::queue<NodeID> bfs_queue;
  for (const NodeID sink : node_status.sink_nodes()) {
    if (labeling[sink] != 0) {
      LOG_WARNING << "Sink condition violated for node " << sink;
      is_valid = false;
    }

    bfs_queue.push(sink);
  }

  std::unordered_set<NodeID> visited;
  while (!bfs_queue.empty()) {
    const NodeID u = bfs_queue.front();
    bfs_queue.pop();

    const NodeID u_label = labeling[u];
    graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
      const bool is_residual_edge = -flow[e] < w;
      if (!is_residual_edge) {
        return;
      }

      const NodeID v_label = labeling[v];
      const bool is_valid_labeling = v_label <= u_label + 1;
      if (is_residual_edge && !is_valid_labeling) {
        LOG_WARNING << "Edge condition violated for edge " << u << " -> " << v;
        is_valid = false;
      }

      if (visited.contains(v)) {
        return;
      }

      bfs_queue.push(v);
      visited.insert(v);
    });
  }

  return is_valid;
}

bool is_valid_flow(
    const CSRGraph &graph, const NodeStatus &node_status, std::span<const EdgeWeight> flow
) {
  if (graph.m() != flow.size()) {
    LOG_WARNING << "Flow size does not equal the number of edges";
    return false;
  }

  struct EdgeHasher {
    [[nodiscard]] std::size_t operator()(const std::pair<NodeID, NodeID> &edge) const noexcept {
      return edge.first ^ (edge.second << 1);
    }
  };
  std::unordered_map<std::pair<NodeID, NodeID>, EdgeID, EdgeHasher> edge_flow_table;
  std::vector<NodeID> flow_excess(graph.n(), 0);

  bool is_valid = true;

  for (const NodeID u : graph.nodes()) {
    graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
      const EdgeWeight e_flow = flow[e];
      if (e_flow > w) {
        LOG_WARNING << "Capacity constraint violated for edge " << u << " -> " << v;
        is_valid = false;
      }

      if (u < v) {
        edge_flow_table[{u, v}] = e;
      } else {
        auto it = edge_flow_table.find({v, u});
        KASSERT(it != edge_flow_table.end());

        const EdgeID e_reverse = it->second;
        const EdgeWeight e_reverse_flow = flow[e_reverse];
        if (e_flow != -e_reverse_flow) {
          LOG_WARNING << "Antisymmetry constraint violated for edge " << v << " <-> " << u;
          is_valid = false;
        }

        edge_flow_table.erase(it);
      }

      if (e_flow > 0) {
        flow_excess[u] -= flow[e];
        flow_excess[v] += flow[e];
      }
    });
  }

  for (const NodeID u : graph.nodes()) {
    if (node_status.is_terminal(u)) {
      continue;
    }

    if (flow_excess[u] != 0) {
      LOG_WARNING << "Conservation constraint violated for node " << u << ' ' << flow_excess[u];
      is_valid = false;
    }
  }

  return is_valid;
}

[[nodiscard]] bool is_valid_preflow(
    const CSRGraph &graph, const NodeStatus &node_status, std::span<const EdgeWeight> flow
) {
  if (graph.m() != flow.size()) {
    LOG_WARNING << "Flow size does not equal the number of edges";
    return false;
  }

  struct EdgeHasher {
    [[nodiscard]] std::size_t operator()(const std::pair<NodeID, NodeID> &edge) const noexcept {
      return edge.first ^ (edge.second << 1);
    }
  };
  std::unordered_map<std::pair<NodeID, NodeID>, EdgeID, EdgeHasher> edge_flow_table;
  std::vector<EdgeWeight> flow_excess(graph.n(), 0);

  bool is_valid = true;

  for (const NodeID u : graph.nodes()) {
    graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
      const EdgeWeight e_flow = flow[e];
      if (e_flow > w) {
        LOG_WARNING << "Capacity constraint violated for edge " << u << " -> " << v;
        is_valid = false;
      }

      if (u < v) {
        edge_flow_table[{u, v}] = e;
      } else {
        auto it = edge_flow_table.find({v, u});
        KASSERT(it != edge_flow_table.end());

        const EdgeID e_reverse = it->second;
        const EdgeWeight e_reverse_flow = flow[e_reverse];
        if (e_flow != -e_reverse_flow) {
          LOG_WARNING << "Antisymmetry constraint violated for edge " << v << " <-> " << u;
          is_valid = false;
        }

        edge_flow_table.erase(it);
      }

      if (e_flow > 0) {
        flow_excess[u] -= flow[e];
        flow_excess[v] += flow[e];
      }
    });
  }

  for (const NodeID u : graph.nodes()) {
    if (node_status.is_terminal(u)) {
      continue;
    }

    if (flow_excess[u] < 0) {
      LOG_WARNING << "Excess constraint violated for node " << u << ' ' << flow_excess[u];
      is_valid = false;
    }
  }

  return is_valid;
}

EdgeWeight
flow_value(const CSRGraph &graph, const NodeStatus &node_status, std::span<const EdgeWeight> flow) {
  EdgeWeight flow_value = 0;

  for (const NodeID sink : node_status.sink_nodes()) {
    graph.neighbors(sink, [&](const EdgeID e, const NodeID v) {
      if (node_status.is_sink(v)) {
        return;
      }

      flow_value += -flow[e];
    });
  }

  return flow_value;
}

bool is_max_flow(
    const CSRGraph &graph, const NodeStatus &node_status, std::span<const EdgeWeight> flow
) {
  if (graph.m() != flow.size()) {
    LOG_WARNING << "Flow size does not equal the number of edges";
    return false;
  }

  std::unordered_set<NodeID> visited;
  std::queue<NodeID> bfs_queue;
  for (const NodeID source : node_status.source_nodes()) {
    bfs_queue.push(source);
    visited.insert(source);
  }

  while (!bfs_queue.empty()) {
    const NodeID u = bfs_queue.front();
    bfs_queue.pop();

    if (node_status.is_sink(u)) {
      return false;
    }

    graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight w) {
      if (visited.contains(v)) {
        return;
      }

      const EdgeWeight e_flow = flow[e];
      if (e_flow < w) {
        bfs_queue.push(v);
        visited.insert(v);
      }
    });
  }

  return true;
}

void print_flow(
    const CSRGraph &graph, const NodeStatus &node_status, std::span<const EdgeWeight> flow
) {
  for (const NodeID u : graph.nodes()) {
    graph.neighbors(u, [&](const EdgeID e, const NodeID v, const EdgeWeight c) {
      std::cout << u << " -> " << v << ": " << flow[e] << "/" << c;

      if (node_status.is_source(u)) {
        std::cout << " (source)";
      } else if (node_status.is_sink(u)) {
        std::cout << " (sink)";
      }

      std::cout << "\n";
    });
  }
}

} // namespace kaminpar::shm::debug
