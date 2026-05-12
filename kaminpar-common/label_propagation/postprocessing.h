/*******************************************************************************
 * Label propagation post-processing helpers.
 *
 * @file:   postprocessing.h
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
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/label_propagation/kernel.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::lp {

namespace detail {

template <bool kMatch, typename Kernel>
void handle_isolated_nodes(
    Kernel &kernel, const typename Kernel::NodeID from, const typename Kernel::NodeID to
) {
  using ClusterID = typename Kernel::ClusterID;
  constexpr ClusterID kInvalidClusterID = std::numeric_limits<ClusterID>::max();
  tbb::enumerable_thread_specific<ClusterID> current_cluster_ets(kInvalidClusterID);

  auto &graph = kernel.graph();
  auto &labels = kernel.labels();
  auto &weights = kernel.weights();

  tbb::parallel_for(
      tbb::blocked_range<typename Kernel::NodeID>(from, std::min(graph.n(), to)),
      [&](tbb::blocked_range<typename Kernel::NodeID> r) {
        ClusterID cluster = current_cluster_ets.local();

        for (typename Kernel::NodeID u = r.begin(); u != r.end(); ++u) {
          if (graph.degree(u) == 0) {
            const ClusterID cu = labels.cluster(u);

            if (cluster != kInvalidClusterID &&
                weights.move_cluster_weight(
                    cu, cluster, weights.cluster_weight(cu), weights.max_cluster_weight(cluster)
                )) {
              labels.move_node(u, cluster);
              if constexpr (kMatch) {
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

template <typename Kernel>
[[nodiscard]] bool
is_considered_for_two_hop_clustering(Kernel &kernel, const typename Kernel::NodeID u) {
  using ClusterID = typename Kernel::ClusterID;
  using ClusterWeight = typename Kernel::ClusterWeight;

  auto &graph = kernel.graph();
  auto &labels = kernel.labels();
  auto &weights = kernel.weights();
  auto &workspace = kernel.workspace();

  if (graph.degree(u) == 0) {
    return false;
  }

  const auto check_cluster_weight = [&](const ClusterID c_u) {
    const ClusterWeight current_weight = weights.cluster_weight(c_u);

    if (current_weight > weights.max_cluster_weight(c_u) / 2 ||
        current_weight != weights.initial_cluster_weight(c_u)) {
      return false;
    }

    return true;
  };

  if (kernel.relabeled()) {
    if (u < workspace.postprocessing.moved.size() && workspace.postprocessing.moved[u]) {
      return false;
    }

    const ClusterID c_u = labels.cluster(u);
    return check_cluster_weight(c_u);
  } else {
    if (u != labels.cluster(u)) {
      return false;
    }

    return check_cluster_weight(u);
  }
}

template <bool kMatch, typename Kernel>
void handle_two_hop_nodes_threadwise_impl(
    Kernel &kernel, const typename Kernel::NodeID from, const typename Kernel::NodeID to
) {
  using NodeID = typename Kernel::NodeID;
  using ClusterID = typename Kernel::ClusterID;
  tbb::enumerable_thread_specific<DynamicFlatMap<ClusterID, NodeID>> matching_map_ets;

  auto &labels = kernel.labels();
  auto &weights = kernel.weights();
  auto &workspace = kernel.workspace();

  auto handle_node = [&](DynamicFlatMap<ClusterID, NodeID> &matching_map, const NodeID u) {
    const ClusterID c_u = labels.cluster(u);
    ClusterID &rep_key = matching_map[workspace.postprocessing.favored_clusters[u]];

    if (rep_key == 0) {
      rep_key = c_u + 1;
    } else {
      const ClusterID rep = rep_key - 1;

      const bool could_move_u_to_rep = weights.move_cluster_weight(
          c_u, rep, weights.cluster_weight(c_u), weights.max_cluster_weight(rep)
      );

      if constexpr (kMatch) {
        KASSERT(could_move_u_to_rep);
        labels.move_node(u, rep);
        rep_key = 0;
      } else {
        if (could_move_u_to_rep) {
          labels.move_node(u, rep);
        } else {
          rep_key = c_u + 1;
        }
      }
    }
  };

  tbb::parallel_for(
      tbb::blocked_range<NodeID>(from, std::min(to, kernel.graph().n()), 512),
      [&](const tbb::blocked_range<NodeID> &r) {
        auto &matching_map = matching_map_ets.local();

        for (NodeID u = r.begin(); u != r.end(); ++u) {
          if (is_considered_for_two_hop_clustering(kernel, u)) {
            handle_node(matching_map, u);
          }
        }
      }
  );
}

template <bool kMatch, typename Kernel>
void handle_two_hop_nodes_impl(
    Kernel &kernel, const typename Kernel::NodeID from, const typename Kernel::NodeID to
) {
  using NodeID = typename Kernel::NodeID;
  using ClusterWeight = typename Kernel::ClusterWeight;

  auto &graph = kernel.graph();
  auto &labels = kernel.labels();
  auto &weights = kernel.weights();
  auto &workspace = kernel.workspace();

  const auto is_considered_for_non_threadwise_two_hop_clustering = [&](const NodeID u) {
    if (graph.degree(u) == 0 || u != labels.cluster(u)) {
      return false;
    }

    const ClusterWeight current_weight = weights.cluster_weight(u);
    return current_weight <= weights.max_cluster_weight(u) / 2 &&
           current_weight == weights.initial_cluster_weight(u);
  };

  tbb::parallel_for(from, std::min(to, graph.n()), [&](const NodeID u) {
    if (is_considered_for_non_threadwise_two_hop_clustering(u)) {
      const NodeID cluster = workspace.postprocessing.favored_clusters[u];
      if (is_considered_for_non_threadwise_two_hop_clustering(cluster) &&
          weights.move_cluster_weight(
              u, cluster, weights.cluster_weight(u), weights.max_cluster_weight(cluster)
          )) {
        labels.move_node(u, cluster);
        kernel.decrement_current_num_clusters();
      }
    } else {
      workspace.postprocessing.favored_clusters[u] = u;
    }
  });

  tbb::parallel_for(from, std::min(to, graph.n()), [&](const NodeID u) {
    if (kernel.should_stop() || !is_considered_for_non_threadwise_two_hop_clustering(u)) {
      return;
    }

    const NodeID c =
        __atomic_load_n(&workspace.postprocessing.favored_clusters[u], __ATOMIC_RELAXED);
    auto &sync = workspace.postprocessing.favored_clusters[c];

    do {
      NodeID cluster = sync;

      if (cluster == c) {
        if (__atomic_compare_exchange_n(
                &sync, &cluster, u, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
            )) {
          break;
        }
        if (cluster == c) {
          continue;
        }
      }

      KASSERT(
          __atomic_load_n(&workspace.postprocessing.favored_clusters[cluster], __ATOMIC_RELAXED) ==
          c
      );

      if constexpr (kMatch) {
        if (__atomic_compare_exchange_n(
                &sync, &cluster, c, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
            )) {
          [[maybe_unused]] const bool success = weights.move_cluster_weight(
              u, cluster, weights.cluster_weight(u), weights.max_cluster_weight(cluster)
          );
          KASSERT(success);

          labels.move_node(u, cluster);
          break;
        }
      } else {
        if (weights.move_cluster_weight(
                u, cluster, weights.cluster_weight(u), weights.max_cluster_weight(cluster)
            )) {
          labels.move_node(u, cluster);
          break;
        } else if (
            __atomic_compare_exchange_n(
                &sync, &cluster, u, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
            )
        ) {
          break;
        }
      }
    } while (true);
  });
}

template <bool kMatch, bool kThreadwise, typename Kernel>
void handle_two_hop_nodes(
    Kernel &kernel, const typename Kernel::NodeID from, const typename Kernel::NodeID to
) {
  KASSERT(kernel.config().selection.track_favored_clusters);

  if constexpr (kThreadwise) {
    handle_two_hop_nodes_threadwise_impl<kMatch>(kernel, from, to);
  } else {
    handle_two_hop_nodes_impl<kMatch>(kernel, from, to);
  }
}

} // namespace detail

template <typename Kernel>
void match_isolated_nodes(
    Kernel &kernel,
    const typename Kernel::NodeID from = 0,
    const typename Kernel::NodeID to = std::numeric_limits<typename Kernel::NodeID>::max()
) {
  detail::handle_isolated_nodes<true>(kernel, from, to);
}

template <typename Kernel>
void cluster_isolated_nodes(
    Kernel &kernel,
    const typename Kernel::NodeID from = 0,
    const typename Kernel::NodeID to = std::numeric_limits<typename Kernel::NodeID>::max()
) {
  detail::handle_isolated_nodes<false>(kernel, from, to);
}

template <typename Kernel>
void match_two_hop_nodes(
    Kernel &kernel,
    const typename Kernel::NodeID from = 0,
    const typename Kernel::NodeID to = std::numeric_limits<typename Kernel::NodeID>::max()
) {
  detail::handle_two_hop_nodes<true, false>(kernel, from, to);
}

template <typename Kernel>
void cluster_two_hop_nodes(
    Kernel &kernel,
    const typename Kernel::NodeID from = 0,
    const typename Kernel::NodeID to = std::numeric_limits<typename Kernel::NodeID>::max()
) {
  detail::handle_two_hop_nodes<false, false>(kernel, from, to);
}

template <typename Kernel>
void match_two_hop_nodes_threadwise(
    Kernel &kernel,
    const typename Kernel::NodeID from = 0,
    const typename Kernel::NodeID to = std::numeric_limits<typename Kernel::NodeID>::max()
) {
  detail::handle_two_hop_nodes<true, true>(kernel, from, to);
}

template <typename Kernel>
void cluster_two_hop_nodes_threadwise(
    Kernel &kernel,
    const typename Kernel::NodeID from = 0,
    const typename Kernel::NodeID to = std::numeric_limits<typename Kernel::NodeID>::max()
) {
  detail::handle_two_hop_nodes<false, true>(kernel, from, to);
}

template <typename Kernel> void relabel_clusters(Kernel &kernel) {
  using NodeID = typename Kernel::NodeID;
  using ClusterID = typename Kernel::ClusterID;

  SCOPED_HEAP_PROFILER("Relabel");
  SCOPED_TIMER("Relabel");

  auto &graph = kernel.graph();
  auto &labels = kernel.labels();
  auto &weights = kernel.weights();
  auto &workspace = kernel.workspace();

  ClusterID num_actual_clusters = kernel.current_num_clusters();
  kernel.set_initial_num_clusters(num_actual_clusters);
  kernel.set_relabeled(true);

  if (workspace.postprocessing.moved.size() < graph.n()) {
    workspace.postprocessing.moved.resize(graph.n());
  }

  StaticArray<ClusterID> mapping(graph.n());
  tbb::parallel_for(tbb::blocked_range<NodeID>(0, graph.n()), [&](const auto &r) {
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      const ClusterID c_u = labels.cluster(u);
      __atomic_store_n(&mapping[c_u], 1, __ATOMIC_RELAXED);

      if (u != c_u) {
        workspace.postprocessing.moved[u] = 1;
      }
    }
  });

  parallel::prefix_sum(mapping.begin(), mapping.end(), mapping.begin());
  KASSERT(num_actual_clusters == mapping[graph.n() - 1]);

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, graph.n()), [&](const auto &r) {
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      labels.move_node(u, mapping[labels.cluster(u)] - 1);

      if (u < workspace.postprocessing.favored_clusters.size()) {
        workspace.postprocessing.favored_clusters[u] =
            mapping[workspace.postprocessing.favored_clusters[u]] - 1;
      }
    }
  });

  weights.reassign_cluster_weights(mapping, num_actual_clusters);
}

} // namespace kaminpar::lp
