/*******************************************************************************
 * Free-standing two-hop clustering and isolated node handling utilities.
 *
 * After label propagation, some nodes may remain in singleton clusters. These
 * utilities merge such nodes by matching/clustering them based on shared
 * "favored clusters" (the best cluster found during LP, even if the node
 * couldn't join it due to weight constraints).
 *
 * @file:   two_hop_clustering.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <limits>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/atomic.h"

namespace kaminpar::lp::two_hop {

// ---------------------------------------------------------------------------
// Isolated nodes (degree-0)
// ---------------------------------------------------------------------------

/*!
 * Handle isolated (degree-0) nodes by either matching or clustering them.
 *
 * When `match` is true, each pair of consecutive isolated nodes is merged.
 * When `match` is false, as many consecutive isolated nodes as possible are
 * merged into a single cluster (limited by max weight).
 *
 * @tparam match If true, match pairwise. If false, cluster greedily.
 */
template <bool match, typename Graph, typename ClusterOps>
void handle_isolated_nodes(
    const Graph &graph,
    ClusterOps &ops,
    const typename Graph::NodeID from,
    const typename Graph::NodeID to
) {
  using NodeID = typename Graph::NodeID;
  using ClusterID = decltype(ops.cluster(std::declval<NodeID>()));
  constexpr ClusterID kInvalidClusterID = std::numeric_limits<ClusterID>::max();

  const NodeID actual_to = std::min(to, graph.n());
  tbb::enumerable_thread_specific<ClusterID> current_cluster_ets(kInvalidClusterID);

  tbb::parallel_for(
      tbb::blocked_range<NodeID>(from, actual_to),
      [&](tbb::blocked_range<NodeID> r) {
        ClusterID cluster = current_cluster_ets.local();

        for (NodeID u = r.begin(); u != r.end(); ++u) {
          if (graph.degree(u) == 0) {
            const ClusterID cu = ops.cluster(u);

            if (cluster != kInvalidClusterID &&
                ops.move_cluster_weight(
                    cu, cluster, ops.cluster_weight(cu), ops.max_cluster_weight(cluster)
                )) {
              ops.move_node(u, cluster);
              if constexpr (match) {
                cluster = kInvalidClusterID;
              }
            } else {
              cluster = cu;
            }
          }
        }

        current_cluster_ets.local() = cluster;
      }
  );
}

template <typename Graph, typename ClusterOps>
void match_isolated_nodes(
    const Graph &graph,
    ClusterOps &ops,
    const typename Graph::NodeID from = 0,
    const typename Graph::NodeID to = std::numeric_limits<typename Graph::NodeID>::max()
) {
  handle_isolated_nodes<true>(graph, ops, from, to);
}

template <typename Graph, typename ClusterOps>
void cluster_isolated_nodes(
    const Graph &graph,
    ClusterOps &ops,
    const typename Graph::NodeID from = 0,
    const typename Graph::NodeID to = std::numeric_limits<typename Graph::NodeID>::max()
) {
  handle_isolated_nodes<false>(graph, ops, from, to);
}

// ---------------------------------------------------------------------------
// Two-hop clustering (thread-wise variant)
// ---------------------------------------------------------------------------

/*!
 * Handle two-hop clustering using thread-local maps for synchronization.
 *
 * Nodes that remain in singleton clusters after LP are merged with other
 * singleton nodes that share the same favored cluster.
 *
 * @tparam match If true, match pairwise. If false, cluster greedily.
 */
template <bool match, typename Graph, typename ClusterOps>
void handle_two_hop_nodes_threadwise(
    const Graph &graph,
    ClusterOps &ops,
    StaticArray<typename Graph::NodeID> &favored_clusters,
    const StaticArray<std::uint8_t> *moved,
    const bool relabeled,
    const typename Graph::NodeID from,
    const typename Graph::NodeID to
) {
  using NodeID = typename Graph::NodeID;
  using ClusterID = decltype(ops.cluster(std::declval<NodeID>()));

  tbb::enumerable_thread_specific<DynamicFlatMap<ClusterID, NodeID>> matching_map_ets;

  auto is_considered_for_two_hop_clustering = [&](const NodeID u) {
    if (graph.degree(u) == 0) {
      return false;
    }

    auto check_cluster_weight = [&](const ClusterID c_u) {
      const auto current_weight = ops.cluster_weight(c_u);
      if (current_weight > ops.max_cluster_weight(c_u) / 2 ||
          current_weight != ops.initial_cluster_weight(c_u)) {
        return false;
      }
      return true;
    };

    if (relabeled) {
      if (moved != nullptr && (*moved)[u]) {
        return false;
      }
      const ClusterID c_u = ops.cluster(u);
      return check_cluster_weight(c_u);
    } else {
      if (u != ops.cluster(u)) {
        return false;
      }
      return check_cluster_weight(u);
    }
  };

  auto handle_node = [&](DynamicFlatMap<ClusterID, NodeID> &matching_map, const NodeID u) {
    const ClusterID c_u = ops.cluster(u);
    ClusterID &rep_key = matching_map[favored_clusters[u]];

    if (rep_key == 0) {
      rep_key = c_u + 1;
    } else {
      const ClusterID rep = rep_key - 1;

      const bool could_move_u_to_rep = ops.move_cluster_weight(
          c_u, rep, ops.cluster_weight(c_u), ops.max_cluster_weight(rep)
      );

      if constexpr (match) {
        KASSERT(could_move_u_to_rep);
        ops.move_node(u, rep);
        rep_key = 0;
      } else {
        if (could_move_u_to_rep) {
          ops.move_node(u, rep);
        } else {
          rep_key = c_u + 1;
        }
      }
    }
  };

  const NodeID actual_to = std::min(to, graph.n());
  tbb::parallel_for(
      tbb::blocked_range<NodeID>(from, actual_to, 512),
      [&](const tbb::blocked_range<NodeID> &r) {
        auto &matching_map = matching_map_ets.local();

        for (NodeID u = r.begin(); u != r.end(); ++u) {
          if (is_considered_for_two_hop_clustering(u)) {
            handle_node(matching_map, u);
          }
        }
      }
  );
}

template <typename Graph, typename ClusterOps>
void match_two_hop_nodes_threadwise(
    const Graph &graph,
    ClusterOps &ops,
    StaticArray<typename Graph::NodeID> &favored_clusters,
    const StaticArray<std::uint8_t> *moved,
    const bool relabeled,
    const typename Graph::NodeID from = 0,
    const typename Graph::NodeID to = std::numeric_limits<typename Graph::NodeID>::max()
) {
  handle_two_hop_nodes_threadwise<true>(graph, ops, favored_clusters, moved, relabeled, from, to);
}

template <typename Graph, typename ClusterOps>
void cluster_two_hop_nodes_threadwise(
    const Graph &graph,
    ClusterOps &ops,
    StaticArray<typename Graph::NodeID> &favored_clusters,
    const StaticArray<std::uint8_t> *moved,
    const bool relabeled,
    const typename Graph::NodeID from = 0,
    const typename Graph::NodeID to = std::numeric_limits<typename Graph::NodeID>::max()
) {
  handle_two_hop_nodes_threadwise<false>(graph, ops, favored_clusters, moved, relabeled, from, to);
}

// ---------------------------------------------------------------------------
// Two-hop clustering (global atomic CAS variant)
// ---------------------------------------------------------------------------

/*!
 * Handle two-hop clustering using global atomic CAS for synchronization.
 *
 * Uses the favored_clusters array as a synchronization mechanism: nodes with
 * the same favored cluster try to merge by atomically claiming a representative.
 *
 * @tparam match If true, match pairwise. If false, cluster greedily.
 */
template <bool match, typename Graph, typename ClusterOps, typename ClusterID>
void handle_two_hop_nodes(
    const Graph &graph,
    ClusterOps &ops,
    StaticArray<ClusterID> &favored_clusters,
    parallel::Atomic<ClusterID> &current_num_clusters,
    const ClusterID desired_num_clusters,
    const typename Graph::NodeID from,
    const typename Graph::NodeID to
) {
  using NodeID = typename Graph::NodeID;

  const NodeID actual_to = std::min(to, graph.n());

  auto is_considered_for_two_hop_clustering = [&](const NodeID u) {
    if (graph.degree(u) == 0) {
      return false;
    } else if (u != ops.cluster(u)) {
      return false;
    } else {
      const auto current_weight = ops.cluster_weight(u);
      if (current_weight > ops.max_cluster_weight(u) / 2 ||
          current_weight != ops.initial_cluster_weight(u)) {
        return false;
      }
    }
    return true;
  };

  auto should_stop = [&] {
    return current_num_clusters <= desired_num_clusters;
  };

  // Pre-pass: move singleton nodes to their favored cluster if the favored cluster is also
  // a singleton (edge case fix).
  tbb::parallel_for(from, actual_to, [&](const NodeID u) {
    if (is_considered_for_two_hop_clustering(u)) {
      const NodeID cluster = favored_clusters[u];
      if (is_considered_for_two_hop_clustering(cluster) &&
          ops.move_cluster_weight(
              u, cluster, ops.cluster_weight(u), ops.max_cluster_weight(cluster)
          )) {
        ops.move_node(u, cluster);
        --current_num_clusters;
      }
    } else {
      favored_clusters[u] = u;
    }
  });

  KASSERT(
      [&] {
        for (NodeID u = from; u < actual_to; ++u) {
          if (favored_clusters[u] >= graph.n()) {
            LOG_WARNING << "favored cluster of node " << u
                        << " out of bounds: " << favored_clusters[u] << " > " << graph.n();
          }
          if (u != favored_clusters[u] && is_considered_for_two_hop_clustering(u) &&
              is_considered_for_two_hop_clustering(favored_clusters[u])) {
            LOG_WARNING << "node " << u << " (degree " << graph.degree(u) << " )"
                        << " is considered for two-hop clustering, but its favored cluster "
                        << favored_clusters[u] << " (degree " << graph.degree(favored_clusters[u])
                        << ") is also considered for two-hop clustering";
            return false;
          }
        }
        return true;
      }(),
      "precondition for two-hop clustering violated: found favored clusters that could be joined",
      assert::heavy
  );

  // Main pass: use atomic CAS on the favored_clusters array to synchronize merging.
  tbb::parallel_for(from, actual_to, [&](const NodeID u) {
    if (should_stop()) {
      return;
    }

    if (!is_considered_for_two_hop_clustering(u)) {
      return;
    }

    const NodeID C = __atomic_load_n(&favored_clusters[u], __ATOMIC_RELAXED);
    auto &sync = favored_clusters[C];

    do {
      NodeID cluster = sync;

      if (cluster == C) {
        if (__atomic_compare_exchange_n(
                &sync, &cluster, u, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
            )) {
          break;
        }
        if (cluster == C) {
          continue;
        }
      }

      KASSERT(__atomic_load_n(&favored_clusters[cluster], __ATOMIC_RELAXED) == C);

      if constexpr (match) {
        if (__atomic_compare_exchange_n(
                &sync, &cluster, C, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
            )) {
          [[maybe_unused]] const bool success = ops.move_cluster_weight(
              u, cluster, ops.cluster_weight(u), ops.max_cluster_weight(cluster)
          );
          KASSERT(
              success,
              "node " << u << " could be matched with node " << cluster << ": "
                      << ops.cluster_weight(u) << " + " << ops.cluster_weight(cluster) << " > "
                      << ops.max_cluster_weight(cluster)
          );

          ops.move_node(u, cluster);
          --current_num_clusters;
          break;
        }
      } else {
        if (ops.move_cluster_weight(
                u, cluster, ops.cluster_weight(u), ops.max_cluster_weight(cluster)
            )) {
          ops.move_node(u, cluster);
          --current_num_clusters;
          break;
        } else if (__atomic_compare_exchange_n(
                       &sync, &cluster, C, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
                   )) {
          break;
        }
      }
    } while (true);
  });
}

template <typename Graph, typename ClusterOps, typename ClusterID>
void match_two_hop_nodes(
    const Graph &graph,
    ClusterOps &ops,
    StaticArray<ClusterID> &favored_clusters,
    parallel::Atomic<ClusterID> &current_num_clusters,
    const ClusterID desired_num_clusters,
    const typename Graph::NodeID from = 0,
    const typename Graph::NodeID to = std::numeric_limits<typename Graph::NodeID>::max()
) {
  handle_two_hop_nodes<true>(
      graph, ops, favored_clusters, current_num_clusters, desired_num_clusters, from, to
  );
}

template <typename Graph, typename ClusterOps, typename ClusterID>
void cluster_two_hop_nodes(
    const Graph &graph,
    ClusterOps &ops,
    StaticArray<ClusterID> &favored_clusters,
    parallel::Atomic<ClusterID> &current_num_clusters,
    const ClusterID desired_num_clusters,
    const typename Graph::NodeID from = 0,
    const typename Graph::NodeID to = std::numeric_limits<typename Graph::NodeID>::max()
) {
  handle_two_hop_nodes<false>(
      graph, ops, favored_clusters, current_num_clusters, desired_num_clusters, from, to
  );
}

} // namespace kaminpar::lp::two_hop
