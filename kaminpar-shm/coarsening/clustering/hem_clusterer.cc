/*******************************************************************************
 * Heavy edge matching for graph coarsening / clustering.
 *
 * @file:   hem_clusterer.cc
 ******************************************************************************/
#include "kaminpar-shm/coarsening/clustering/hem_clusterer.h"

#include <algorithm>
#include <atomic>

#include <tbb/parallel_for.h>

#include "kaminpar-shm/datastructures/graph.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

namespace {

SET_DEBUG(false);

[[nodiscard]] bool
accepts_community(const std::span<const NodeID> communities, const NodeID u, const NodeID v) {
  return communities.empty() || communities[u] == communities[v];
}

template <typename Graph>
[[nodiscard]] bool accepts_weight(
    const Graph &graph, const NodeWeight max_cluster_weight, const NodeID u, const NodeID v
) {
  return graph.node_weight(u) <= max_cluster_weight - graph.node_weight(v);
}

[[nodiscard]] bool is_better_match(
    const EdgeWeight weight, const NodeID node, const EdgeWeight best_weight, const NodeID best_node
) {
  return weight > best_weight || (weight == best_weight && node < best_node);
}

[[nodiscard]] NodeID max_num_matches(const NodeID n, const NodeID desired_cluster_count) {
  if (desired_cluster_count == 0) {
    return n;
  }
  if (desired_cluster_count >= n) {
    return 0;
  }
  return n - desired_cluster_count;
}

[[nodiscard]] bool try_reserve_match(
    std::atomic<NodeID> &num_matches, const NodeID n, const NodeID desired_cluster_count
) {
  const NodeID max_matches = max_num_matches(n, desired_cluster_count);
  NodeID current = num_matches.load(std::memory_order_relaxed);
  while (current < max_matches) {
    if (num_matches.compare_exchange_weak(
            current, current + 1, std::memory_order_acq_rel, std::memory_order_relaxed
        )) {
      return true;
    }
  }

  return false;
}

void release_match_reservation(std::atomic<NodeID> &num_matches) {
  num_matches.fetch_sub(1, std::memory_order_acq_rel);
}

[[nodiscard]] bool should_stop(
    const std::atomic<NodeID> &num_matches, const NodeID n, const NodeID desired_cluster_count
) {
  return num_matches.load(std::memory_order_relaxed) >= max_num_matches(n, desired_cluster_count);
}

[[nodiscard]] bool is_unmatched(const StaticArray<NodeID> &matching, const NodeID u) {
  return __atomic_load_n(&matching[u], __ATOMIC_RELAXED) == kInvalidNodeID;
}

template <typename Graph>
[[nodiscard]] bool try_match(
    StaticArray<NodeID> &matching,
    std::atomic<NodeID> &num_matches,
    const Graph &graph,
    const std::span<const NodeID> communities,
    const NodeWeight max_cluster_weight,
    const NodeID desired_cluster_count,
    const NodeID u,
    const NodeID v
) {
  KASSERT(u < graph.n());
  KASSERT(v < graph.n());

  if (u == v || !accepts_community(communities, u, v) ||
      !accepts_weight(graph, max_cluster_weight, u, v)) {
    return false;
  }

  if (!try_reserve_match(num_matches, graph.n(), desired_cluster_count)) {
    return false;
  }

  const NodeID leader = std::min(u, v);
  const NodeID first = leader;
  const NodeID second = std::max(u, v);

  NodeID expected = kInvalidNodeID;
  if (!__atomic_compare_exchange_n(
          &matching[first], &expected, leader, true, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
      )) {
    release_match_reservation(num_matches);
    return false;
  }

  expected = kInvalidNodeID;
  if (!__atomic_compare_exchange_n(
          &matching[second], &expected, leader, true, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
      )) {
    __atomic_store_n(&matching[first], kInvalidNodeID, __ATOMIC_SEQ_CST);
    release_match_reservation(num_matches);
    return false;
  }

  return true;
}

template <typename Graph>
void compute_heavy_edge_matching(
    StaticArray<NodeID> &matching,
    StaticArray<NodeID> &favored,
    StaticArray<NodeID> &best,
    std::atomic<NodeID> &num_matches,
    const Graph &graph,
    const std::span<const NodeID> communities,
    const NodeWeight max_cluster_weight,
    const NodeID desired_cluster_count
) {
  SCOPED_TIMER("Heavy edge matching");

  graph.pfor_nodes([&](const NodeID u) {
    NodeID favored_neighbor = u;
    EdgeWeight favored_weight = 0;
    bool found_favored_neighbor = false;

    NodeID best_neighbor = kInvalidNodeID;
    EdgeWeight best_weight = 0;
    bool found_best_neighbor = false;

    graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
      if (!accepts_community(communities, u, v)) {
        return;
      }

      if (!found_favored_neighbor || is_better_match(weight, v, favored_weight, favored_neighbor)) {
        favored_neighbor = v;
        favored_weight = weight;
        found_favored_neighbor = true;
      }

      if (!accepts_weight(graph, max_cluster_weight, u, v)) {
        return;
      }

      if (!found_best_neighbor || is_better_match(weight, v, best_weight, best_neighbor)) {
        best_neighbor = v;
        best_weight = weight;
        found_best_neighbor = true;
      }
    });

    favored[u] = favored_neighbor;
    best[u] = found_best_neighbor ? best_neighbor : kInvalidNodeID;
  });

  graph.pfor_nodes([&](const NodeID u) {
    if (should_stop(num_matches, graph.n(), desired_cluster_count) || !is_unmatched(matching, u)) {
      return;
    }

    const NodeID best_neighbor = best[u];
    if (best_neighbor != kInvalidNodeID) {
      (void)try_match(
          matching,
          num_matches,
          graph,
          communities,
          max_cluster_weight,
          desired_cluster_count,
          u,
          best_neighbor
      );
    }
  });
}

template <typename Graph>
void compute_two_hop_matching(
    StaticArray<NodeID> &matching,
    StaticArray<NodeID> &favored,
    StaticArray<NodeID> &pending,
    std::atomic<NodeID> &num_matches,
    const Graph &graph,
    const std::span<const NodeID> communities,
    const NodeWeight max_cluster_weight,
    const NodeID desired_cluster_count
) {
  SCOPED_TIMER("Two-hop matching");

  graph.pfor_nodes([&](const NodeID u) { pending[u] = kInvalidNodeID; });

  graph.pfor_nodes([&](const NodeID u) {
    if (should_stop(num_matches, graph.n(), desired_cluster_count) || !is_unmatched(matching, u) ||
        graph.degree(u) == 0) {
      return;
    }

    const NodeID center = __atomic_load_n(&favored[u], __ATOMIC_RELAXED);
    if (center == u || center >= graph.n()) {
      return;
    }

    while (!should_stop(num_matches, graph.n(), desired_cluster_count)) {
      NodeID mate = __atomic_load_n(&pending[center], __ATOMIC_RELAXED);
      if (mate == kInvalidNodeID) {
        if (__atomic_compare_exchange_n(
                &pending[center], &mate, u, true, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
            )) {
          return;
        }
        continue;
      }

      if (mate == u) {
        return;
      }

      if (!is_unmatched(matching, mate) || !accepts_community(communities, u, mate) ||
          !accepts_weight(graph, max_cluster_weight, u, mate)) {
        if (__atomic_compare_exchange_n(
                &pending[center], &mate, u, true, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
            )) {
          return;
        }
        continue;
      }

      if (__atomic_compare_exchange_n(
              &pending[center], &mate, kInvalidNodeID, true, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
          )) {
        (void)try_match(
            matching,
            num_matches,
            graph,
            communities,
            max_cluster_weight,
            desired_cluster_count,
            u,
            mate
        );
        return;
      }
    }
  });
}

template <typename Graph>
void finalize_matching(StaticArray<NodeID> &matching, const Graph &graph) {
  graph.pfor_nodes([&](const NodeID u) {
    const NodeID partner = __atomic_load_n(&matching[u], __ATOMIC_RELAXED);
    matching[u] = partner == kInvalidNodeID ? u : partner;
  });
}

template <typename Graph>
[[nodiscard]] bool validate_matching(const StaticArray<NodeID> &matching, const Graph &graph) {
  for (const NodeID u : graph.nodes()) {
    const NodeID cluster = matching[u];
    KASSERT(cluster < graph.n(), "invalid HEM cluster " << cluster << " for node " << u);
    KASSERT(
        matching[cluster] == cluster,
        "HEM cluster " << cluster << " is not represented by its leader"
    );
  }

  return true;
}

} // namespace

HEMClustering::HEMClustering([[maybe_unused]] const CoarseningContext &c_ctx) {}

void HEMClustering::set_max_cluster_weight(const NodeWeight max_cluster_weight) {
  _max_cluster_weight = max_cluster_weight;
}

void HEMClustering::set_desired_cluster_count(const NodeID count) {
  _desired_cluster_count = count;
}

void HEMClustering::set_communities(const std::span<const NodeID> communities) {
  _communities = communities;
}

void HEMClustering::compute_clustering(
    StaticArray<NodeID> &clustering,
    const Graph &graph,
    [[maybe_unused]] const bool free_memory_afterwards
) {
  SCOPED_HEAP_PROFILER("Heavy Edge Matching");
  SCOPED_TIMER("Heavy Edge Matching");

  reified(graph, [&](const auto &graph) {
    StaticArray<NodeID> favored(graph.n(), static_array::noinit);
    StaticArray<NodeID> best(graph.n(), static_array::noinit);
    StaticArray<NodeID> pending(graph.n(), static_array::noinit);
    std::atomic<NodeID> num_matches{0};

    graph.pfor_nodes([&](const NodeID u) {
      clustering[u] = kInvalidNodeID;
      favored[u] = u;
    });

    compute_heavy_edge_matching(
        clustering,
        favored,
        best,
        num_matches,
        graph,
        _communities,
        _max_cluster_weight,
        _desired_cluster_count
    );
    compute_two_hop_matching(
        clustering,
        favored,
        pending,
        num_matches,
        graph,
        _communities,
        _max_cluster_weight,
        _desired_cluster_count
    );
    finalize_matching(clustering, graph);

    KASSERT(validate_matching(clustering, graph), "matching in inconsistent state", assert::heavy);
  });
}

} // namespace kaminpar::shm
