/*******************************************************************************
 * Standalone in-order iteration strategy for label propagation.
 *
 * Visits nodes in their natural order (0, 1, 2, ...) using TBB parallel_for.
 * This is a standalone, independently testable component — no CRTP.
 *
 * The interface mirrors ChunkRandomIterator<Config>: a caller prepares the
 * iteration range once per graph with prepare(), then calls iterate() once per
 * LP iteration and clear() when the graph changes.
 *
 * @file:   in_order_iteration.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <algorithm>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/label_propagation/config.h"

#include "kaminpar-common/parallel/atomic.h"

namespace kaminpar::lp {

template <typename Config> class InOrderIterator {
public:
  using NodeID = shm::NodeID;

  /*!
   * Store the iteration range for the current graph. Idempotent — calling
   * prepare() again with the same graph is safe and does nothing.
   *
   * @param graph      The graph (used only to clamp `to` at graph.n()).
   * @param from       First node in the iteration range.
   * @param to         One past the last node in the iteration range.
   * @param max_degree Ignored (kept for interface parity with ChunkRandomIterator).
   */
  template <typename Graph>
  void prepare(const Graph &graph, const NodeID from, const NodeID to, const NodeID /* max_degree */) {
    _from = from;
    _to = std::min(to, graph.n());
  }

  /*!
   * Reset the stored range so that the next prepare() call re-initializes it.
   * Call this when the underlying graph changes between outer LP calls.
   */
  void clear() {
    _from = 0;
    _to = 0;
  }

  /*!
   * Iterate over nodes in the range stored by prepare(), calling `handler(u)`
   * for each active node with degree < max_degree. The handler returns
   * `std::pair<bool, bool>` where the first element indicates whether the node
   * was moved and the second whether an empty cluster was created.
   *
   * `should_stop()` is checked approximately every Config::kMinChunkSize edges
   * of work for early termination.
   *
   * @return The total number of nodes that were moved.
   */
  template <typename Graph, typename NodeHandler, typename ShouldStopFn, typename IsActiveFn, typename ClusterID>
  NodeID iterate(
      const Graph &graph,
      const NodeID max_degree,
      NodeHandler &&handler,
      ShouldStopFn &&should_stop,
      IsActiveFn &&is_active,
      parallel::Atomic<ClusterID> &current_num_clusters
  ) {
    tbb::enumerable_thread_specific<NodeID> num_moved_nodes_ets;

    tbb::parallel_for(tbb::blocked_range<NodeID>(_from, _to), [&](const auto &r) {
      shm::EdgeID work_since_update = 0;
      NodeID num_removed_clusters = 0;
      auto &num_moved_nodes = num_moved_nodes_ets.local();

      for (NodeID u = r.begin(); u != r.end(); ++u) {
        if (graph.degree(u) > max_degree) {
          continue;
        }

        if (!is_active(u)) {
          continue;
        }

        if (work_since_update > Config::kMinChunkSize) {
          if (should_stop()) {
            return;
          }

          current_num_clusters -= num_removed_clusters;
          work_since_update = 0;
          num_removed_clusters = 0;
        }

        const auto [moved_node, emptied_cluster] = handler(u);
        work_since_update += graph.degree(u);
        if (moved_node) {
          ++num_moved_nodes;
        }
        if (emptied_cluster) {
          ++num_removed_clusters;
        }
      }
    });

    return num_moved_nodes_ets.combine(std::plus{});
  }

private:
  NodeID _from = 0;
  NodeID _to = 0;
};

} // namespace kaminpar::lp
