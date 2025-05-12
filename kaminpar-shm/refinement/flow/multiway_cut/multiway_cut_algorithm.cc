#include "kaminpar-shm/refinement/flow/multiway_cut/multiway_cut_algorithm.h"

#include <queue>
#include <unordered_set>

namespace kaminpar::shm {

[[nodiscard]] MultiwayCutAlgorithm::Result MultiwayCutAlgorithm::compute(
    [[maybe_unused]] const PartitionedCSRGraph &p_graph,
    const CSRGraph &graph,
    const std::vector<std::unordered_set<NodeID>> &terminal_sets
) {
  return compute(graph, terminal_sets);
}

namespace debug {

bool is_valid_multiway_cut(
    const CSRGraph &graph,
    const std::vector<std::unordered_set<NodeID>> &terminal_sets,
    const std::unordered_set<EdgeID> &cut_edges
) {
  std::unordered_set<NodeID> other_terminals;
  for (const std::unordered_set<NodeID> &terminals : terminal_sets) {
    for (const NodeID terminal : terminals) {
      other_terminals.insert(terminal);
    }
  }

  std::unordered_set<NodeID> visited;
  std::queue<NodeID> bfs_queue;
  for (const std::unordered_set<NodeID> &terminals : terminal_sets) {
    for (NodeID terminal : terminals) {
      other_terminals.erase(terminal);
    }

    visited.clear();
    for (const NodeID terminal : terminals) {
      visited.insert(terminal);
      bfs_queue.push(terminal);
    }

    while (!bfs_queue.empty()) {
      const NodeID u = bfs_queue.front();
      bfs_queue.pop();

      bool is_any_other_terminal_reachable = false;
      graph.neighbors(u, [&](const EdgeID e, const NodeID v) {
        if (!visited.contains(v) && !cut_edges.contains(e)) {
          if (other_terminals.contains(v)) {
            is_any_other_terminal_reachable = true;
            return true;
          }

          visited.insert(v);
          bfs_queue.push(v);
        }

        return false;
      });

      if (is_any_other_terminal_reachable) {
        return false;
      }
    }

    for (const NodeID terminal : terminals) {
      other_terminals.insert(terminal);
    }
  }

  return true;
}

} // namespace debug

} // namespace kaminpar::shm
