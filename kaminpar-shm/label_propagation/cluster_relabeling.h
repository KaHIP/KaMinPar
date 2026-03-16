/*******************************************************************************
 * Free-standing utility for relabeling clusters to consecutive IDs.
 *
 * After label propagation, cluster IDs may be sparse (e.g., only a subset of
 * [0, n) are actually used). This utility compacts them into [0, num_clusters).
 *
 * @file:   cluster_relabeling.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/parallel/algorithm.h"

namespace kaminpar::lp {

/*!
 * Relabels the clusters assigned by label propagation such that cluster IDs become consecutive in
 * [0, num_actual_clusters). Optionally updates a favored-clusters array and records which nodes
 * moved (for two-hop clustering).
 *
 * @tparam NodeID The node ID type.
 * @tparam ClusterID The cluster ID type.
 * @tparam ClusterOps A type providing cluster(u) and move_node(u, c) and
 *         reassign_cluster_weights(mapping, n).
 *
 * @param n Number of nodes.
 * @param ops Cluster operations (must provide cluster(u), move_node(u, c),
 *            reassign_cluster_weights(mapping, num_new_clusters)).
 * @param num_actual_clusters The number of non-empty clusters.
 * @param favored_clusters If non-null, relabel the favored clusters too.
 * @param moved If non-null, mark nodes that joined another cluster (u != cluster(u)).
 *
 * @return The number of actual clusters after relabeling (same as num_actual_clusters).
 */
template <typename NodeID, typename ClusterID, typename ClusterOps>
ClusterID relabel_clusters(
    const NodeID n,
    ClusterOps &ops,
    const ClusterID num_actual_clusters,
    StaticArray<ClusterID> *favored_clusters,
    StaticArray<std::uint8_t> *moved
) {
  // Compute a mapping from old cluster IDs to new cluster IDs.
  StaticArray<ClusterID> mapping(n);
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, n), [&](const auto &r) {
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      const ClusterID c_u = ops.cluster(u);
      __atomic_store_n(&mapping[c_u], 1, __ATOMIC_RELAXED);

      if (moved != nullptr) {
        if (u != c_u) {
          (*moved)[u] = 1;
        }
      }
    }
  });

  parallel::prefix_sum(mapping.begin(), mapping.end(), mapping.begin());

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, n), [&](const auto &r) {
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      // Relabel the cluster stored for each node.
      ops.move_node(u, mapping[ops.cluster(u)] - 1);

      // Relabel the clusters stored in the favored clusters vector.
      if (favored_clusters != nullptr) {
        (*favored_clusters)[u] = mapping[(*favored_clusters)[u]] - 1;
      }
    }
  });

  // Reassign the cluster weights such that they match the new cluster IDs.
  ops.reassign_cluster_weights(mapping, num_actual_clusters);

  return num_actual_clusters;
}

} // namespace kaminpar::lp
