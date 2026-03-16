/*******************************************************************************
 * Standalone in-order iteration strategy for label propagation.
 *
 * Visits nodes in their natural order (0, 1, 2, ...) using TBB parallel_for.
 * This is a standalone, independently testable component — no CRTP.
 *
 * @file:   in_order_iteration.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <limits>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-common/parallel/atomic.h"

namespace kaminpar::lp {

class InOrderIterator {
public:
  /*!
   * Iterate over nodes in [from, to) in natural order, calling `handler(u)` for each node.
   * The handler returns `std::pair<bool, bool>` where the first element indicates whether the
   * node was moved and the second whether an empty cluster was created.
   *
   * @return The total number of nodes that were moved.
   */
  template <
      typename NodeID,
      typename EdgeID,
      typename ClusterID,
      typename Graph,
      typename NodeHandler,
      typename ShouldStopFn,
      typename IsActiveFn>
  static NodeID iterate(
      const Graph &graph,
      const NodeID from,
      const NodeID to,
      const NodeID max_degree,
      const NodeID min_chunk_size,
      NodeHandler &&handler,
      ShouldStopFn &&should_stop,
      IsActiveFn &&is_active,
      parallel::Atomic<ClusterID> &current_num_clusters
  ) {
    const NodeID actual_to = std::min(to, graph.n());
    tbb::enumerable_thread_specific<NodeID> num_moved_nodes_ets;

    tbb::parallel_for(
        tbb::blocked_range<NodeID>(from, actual_to), [&](const auto &r) {
          EdgeID work_since_update = 0;
          NodeID num_removed_clusters = 0;

          auto &num_moved_nodes = num_moved_nodes_ets.local();

          for (NodeID u = r.begin(); u != r.end(); ++u) {
            if (graph.degree(u) > max_degree) {
              continue;
            }

            if (!is_active(u)) {
              continue;
            }

            if (work_since_update > min_chunk_size) {
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
        }
    );

    return num_moved_nodes_ets.combine(std::plus{});
  }
};

} // namespace kaminpar::lp
