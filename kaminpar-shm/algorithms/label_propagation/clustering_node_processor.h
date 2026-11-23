/*******************************************************************************
 * Node-level semantics for label-propagation clustering.
 *
 * @file:   clustering_node_processor.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <type_traits>

#include <tbb/concurrent_vector.h>

#include "kaminpar-shm/algorithms/label_propagation/clustering_selector.h"
#include "kaminpar-shm/algorithms/label_propagation/neighborhood_ratings.h"
#include "kaminpar-shm/algorithms/label_propagation/node_processing.h"
#include "kaminpar-shm/algorithms/label_propagation/parallel_rating_map.h"
#include "kaminpar-shm/algorithms/label_propagation/rating_map_pool.h"
#include "kaminpar-shm/datastructures/csr_graph.h"

#include "kaminpar-common/datastructures/cache_aligned_vector.h"
#include "kaminpar-common/inline.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm::lp {

template <typename Graph> class ClusteringNodeProcessor {
  static constexpr EdgeID kStopCheckWork = 1024;

public:
  using RatingMaps = RatingMapPool<NodeID, EdgeWeight>;
  using RatingMap = typename RatingMaps::RatingMap;

  struct LocalWorker {
    RatingMap &ratings;
    ScalableVector<NodeID> &ties;
    ScalableVector<NodeID> &favored_ties;
    Random &random;
    RoundStats &stats;
    NodeID removed_clusters = 0;
    EdgeID work_since_stop_check = 0;
  };

  ClusteringNodeProcessor(
      const Graph &graph,
      ClusteringState &state,
      RatingMaps &rating_maps,
      ParallelRatingMap<NodeID, EdgeWeight> &parallel_ratings,
      tbb::concurrent_vector<NodeID> &deferred_nodes,
      CacheAlignedVector<ClusteringSelector::RatedSelection> &local_selections,
      RoundStatistics &statistics
  )
      : _graph(graph),
        _state(state),
        _state_view(state.view()),
        _rating_maps(rating_maps),
        _parallel_ratings(parallel_ratings),
        _deferred_nodes(deferred_nodes),
        _local_selections(local_selections),
        _statistics(statistics),
        _selector(_state_view) {}

  [[nodiscard]] bool should_stop() const {
    return _state.should_stop();
  }

  [[nodiscard]] bool should_visit(const NodeID u) const {
    return _state_view.is_active(u);
  }

  [[nodiscard]] bool should_stop(LocalWorker &local, const NodeID u) {
    local.work_since_stop_check += _graph.degree(u);
    if (local.work_since_stop_check <= kStopCheckWork) {
      return false;
    }

    finish_work_unit(local);
    local.work_since_stop_check = 0;
    return should_stop();
  }

  [[nodiscard]] LocalWorker make_local_worker(Random &random) {
    auto local = _rating_maps.local();
    return {
        local.ratings,
        local.ties,
        local.favored_ties,
        random,
        _statistics.local(),
    };
  }

  KAMINPAR_INLINE void visit(const NodeID u, LocalWorker &local) {
    visit_impl<false, true>(u, local);
  }

  KAMINPAR_INLINE void visit_final(const NodeID u, LocalWorker &local) {
    visit_impl<false, false>(u, local);
  }

  void visit_first_phase(const NodeID u, LocalWorker &local) {
    visit_impl<true, true>(u, local);
  }

  void visit_first_phase_final(const NodeID u, LocalWorker &local) {
    visit_impl<true, false>(u, local);
  }

  void finish_work_unit(LocalWorker &local) {
    _state.remove_empty_clusters(local.removed_clusters);
    local.removed_clusters = 0;
  }

  void process_deferred_nodes(const bool activate_neighbors = true) {
    _parallel_ratings.ensure_capacity(_graph.n());
    auto &stats = _statistics.local();
    auto local = _rating_maps.local();
    Random &random = Random::instance();

    for (const NodeID u : _deferred_nodes) {
      const NodeWeight u_weight = _graph.node_weight(u);
      const NodeID from = _state_view.cluster(u);
      const NodeWeight from_weight = _state_view.cluster_weight(from);
      const bool store_favored_cluster =
          u_weight == from_weight && from_weight <= _state_view.max_cluster_weight(from) / 2;
      _parallel_ratings.accumulate(
          _graph,
          u,
          _rating_maps,
          [&](const NodeID v) { return _state_view.cluster(v); },
          [](const NodeID) { return true; }
      );
      if (activate_neighbors) {
        _state_view.deactivate(u);
      }

      _parallel_ratings.for_each_partition_and_reset([&](const std::size_t i, auto &&entries) {
        auto scratch = _rating_maps.local();
        auto selection = _selector.select_rated(
            from,
            u_weight,
            store_favored_cluster,
            entries,
            Random::instance(),
            scratch.ties,
            scratch.favored_ties
        );
        if (store_favored_cluster) {
          selection.favored_rating = _parallel_ratings[selection.favored_cluster];
        }
        _local_selections[i] = selection;
      });

      ClusteringSelector::RatedSelection selection{
          .cluster = from,
          .rating = 0,
          .favored_cluster = from,
          .favored_rating = 0,
      };
      UniformTieSet best_ties(local.ties);
      UniformTieSet favored_ties(local.favored_ties);
      for (auto &candidate : _local_selections) {
        if (candidate.rating > selection.rating) {
          selection.cluster = candidate.cluster;
          selection.rating = candidate.rating;
          best_ties.replace_with(candidate.cluster);
        } else if (candidate.rating == selection.rating) {
          best_ties.add(candidate.cluster);
        }

        if (store_favored_cluster) {
          if (candidate.favored_rating > selection.favored_rating) {
            selection.favored_cluster = candidate.favored_cluster;
            selection.favored_rating = candidate.favored_rating;
            favored_ties.replace_with(candidate.favored_cluster);
          } else if (candidate.favored_rating == selection.favored_rating) {
            favored_ties.add(candidate.favored_cluster);
          }
        }

        candidate.rating = -1;
        candidate.favored_rating = -1;
      }
      selection.cluster = best_ties.select_or(selection.cluster, random);
      if (store_favored_cluster) {
        selection.favored_cluster = favored_ties.select_or(selection.favored_cluster, random);
      }
      best_ties.clear();
      favored_ties.clear();

      remember_favored_cluster(u, from, selection, store_favored_cluster);
      const MoveResult result =
          activate_neighbors
              ? _state_view.commit<true, true>(_graph, u, from, selection.cluster, u_weight)
              : _state_view.commit<true, false>(_graph, u, from, selection.cluster, u_weight);
      if (result.moved) {
        ++stats.moved;
      }
      if (result.emptied_cluster) {
        _state.remove_empty_clusters(1);
      }
    }
    _deferred_nodes.clear();
  }

private:
  template <bool kDefer, bool kTrackActive = true>
  KAMINPAR_INLINE void visit_impl(const NodeID u, LocalWorker &local) {
    if constexpr (std::is_same_v<std::decay_t<Graph>, CSRGraph>) {
      const EdgeID *__restrict__ g_nodes = _graph.raw_nodes().data();
      const NodeID *__restrict__ g_edges = _graph.raw_edges().data();
      const EdgeID e_begin = g_nodes[u];
      const EdgeID e_end = g_nodes[u + 1];
      const EdgeID degree = e_end - e_begin;

      if (degree == 0) [[unlikely]] {
        if constexpr (kTrackActive) {
          _state_view.deactivate(u);
        }
        return;
      }

      const NodeWeight u_weight = _graph.node_weight(u);
      const NodeID from = _state_view.cluster(u);

      if (degree == 1) {
        const NodeID v = g_edges[e_begin];
        const NodeID c_v = _state_view.cluster(v);
        if constexpr (kTrackActive) {
          _state_view.deactivate(u);
        }
        if (c_v != from) {
          const bool accepts =
              (!_state_view.has_communities() ||
               _state_view.accepts_community_unchecked(from, c_v));
          const MoveResult result =
              accepts ? _state_view.commit<false, kTrackActive>(_graph, u, from, c_v, u_weight)
                      : MoveResult{};
          if (result.moved) {
            record_move(result, local);
            return;
          }
        }
        const NodeWeight from_weight = _state_view.cluster_weight(from);
        if (u_weight == from_weight && from_weight <= _state_view.max_cluster_weight(from) / 2) {
          _state_view.set_favored_cluster(u, c_v);
        }
        return;
      }

      const std::size_t upper_bound =
          std::min<std::size_t>(static_cast<std::size_t>(degree), _graph.n());
      const std::size_t map_bound =
          kDefer ? std::min(upper_bound, ParallelRatingMap<NodeID, EdgeWeight>::kFlushThreshold)
                 : upper_bound;

      const auto target = local.ratings.execute(map_bound, [&](auto &ratings) -> std::optional<NodeID> {
        if (accumulate_csr<kDefer>(u, e_begin, e_end, upper_bound, g_edges, ratings)) [[unlikely]] {
          ratings.clear();
          _deferred_nodes.push_back(u);
          return std::nullopt;
        }

        if constexpr (kTrackActive) {
          _state_view.deactivate(u);
        }
        const NodeID best = _selector.select_cluster(from, u_weight, ratings, local.random, local.ties);
        if (best == from) {
          const NodeWeight from_weight = _state_view.cluster_weight(from);
          if (u_weight == from_weight && from_weight <= _state_view.max_cluster_weight(from) / 2) {
            const NodeID favored =
                _selector.select_favored_cluster(from, ratings, local.random, local.favored_ties);
            _state_view.set_favored_cluster(u, favored);
          }
        }
        ratings.clear();
        return best;
      });

      if (target && *target != from) {
        record_move(_state_view.commit<false, kTrackActive>(_graph, u, from, *target, u_weight), local);
      }
    } else {
      const NodeID degree = _graph.degree(u);
      const NodeWeight u_weight = _graph.node_weight(u);
      const NodeID from = _state_view.cluster(u);
      const std::size_t upper_bound = std::min<NodeID>(degree, _graph.n());
      const std::size_t map_bound =
          kDefer ? std::min(upper_bound, ParallelRatingMap<NodeID, EdgeWeight>::kFlushThreshold)
                 : upper_bound;

      const auto target = local.ratings.execute(map_bound, [&](auto &ratings) -> std::optional<NodeID> {
        if constexpr (kDefer) {
          if (NeighborhoodRatings::accumulate_with_capacity(
                  _graph,
                  u,
                  ratings,
                  map_bound,
                  ParallelRatingMap<NodeID, EdgeWeight>::kFlushThreshold,
                  [&](const NodeID v) { return _state_view.cluster(v); },
                  [](const NodeID) { return true; }
              )) [[unlikely]] {
            ratings.clear();
            _deferred_nodes.push_back(u);
            return std::nullopt;
          }
        } else {
          NeighborhoodRatings::accumulate(
              _graph,
              u,
              ratings,
              [&](const NodeID v) { return _state_view.cluster(v); },
              [](const NodeID) { return true; }
          );
        }

        if constexpr (kTrackActive) {
          _state_view.deactivate(u);
        }
        const NodeID best = _selector.select_cluster(from, u_weight, ratings, local.random, local.ties);
        if (best == from) {
          const NodeWeight from_weight = _state_view.cluster_weight(from);
          if (u_weight == from_weight && from_weight <= _state_view.max_cluster_weight(from) / 2) {
            const NodeID favored =
                _selector.select_favored_cluster(from, ratings, local.random, local.favored_ties);
            _state_view.set_favored_cluster(u, favored);
          }
        }
        ratings.clear();
        return best;
      });

      if (target && *target != from) {
        record_move(_state_view.commit<false, kTrackActive>(_graph, u, from, *target, u_weight), local);
      }
    }
  }

  template <bool kDefer, typename Map>
  [[nodiscard]] KAMINPAR_INLINE bool accumulate_csr(
      const NodeID,
      const EdgeID e_begin,
      const EdgeID e_end,
      const std::size_t upper_bound,
      const NodeID *__restrict__ g_edges,
      Map &ratings
  ) const {
    const bool check_full =
        kDefer && upper_bound >= ParallelRatingMap<NodeID, EdgeWeight>::kFlushThreshold;
    const bool edge_weighted = _graph.is_edge_weighted();
    const EdgeWeight *__restrict__ g_ew = edge_weighted ? _graph.raw_edge_weights().data() : nullptr;

    if (check_full) [[unlikely]] {
      constexpr std::size_t kThreshold = ParallelRatingMap<NodeID, EdgeWeight>::kFlushThreshold;
      if constexpr (requires { ratings.accumulator(); }) {
        auto acc = ratings.accumulator();
        for (EdgeID e = e_begin; e < e_end; ++e) {
          acc.add(_state_view.cluster(g_edges[e]), edge_weighted ? g_ew[e] : 1);
          if (acc.size() >= kThreshold) [[unlikely]] {
            return true;
          }
        }
      } else {
        for (EdgeID e = e_begin; e < e_end; ++e) {
          ratings[_state_view.cluster(g_edges[e])] += edge_weighted ? g_ew[e] : 1;
          if (ratings.size() >= kThreshold) [[unlikely]] {
            return true;
          }
        }
      }
      return false;
    }

    if constexpr (requires { ratings.accumulator(); }) {
      auto acc = ratings.accumulator();
      if (edge_weighted) {
        for (EdgeID e = e_begin; e < e_end; ++e) {
          acc.add(_state_view.cluster(g_edges[e]), g_ew[e]);
        }
      } else {
        for (EdgeID e = e_begin; e < e_end; ++e) {
          acc.add(_state_view.cluster(g_edges[e]), 1);
        }
      }
    } else {
      if (edge_weighted) {
        for (EdgeID e = e_begin; e < e_end; ++e) {
          ratings[_state_view.cluster(g_edges[e])] += g_ew[e];
        }
      } else {
        for (EdgeID e = e_begin; e < e_end; ++e) {
          ratings[_state_view.cluster(g_edges[e])] += 1;
        }
      }
    }
    return false;
  }

  template <typename Selection>
  void remember_favored_cluster(
      const NodeID u,
      const NodeID from,
      const Selection &selection,
      const bool store_favored_cluster
  ) {
    if (selection.cluster == from && store_favored_cluster) {
      _state_view.set_favored_cluster(u, selection.favored_cluster);
    }
  }

  static void record_move(const MoveResult result, LocalWorker &local) {
    if (result.moved) {
      ++local.stats.moved;
    }
    if (result.emptied_cluster) {
      ++local.removed_clusters;
    }
  }

  const Graph &_graph;
  ClusteringState &_state;
  ClusteringStateView _state_view;
  RatingMaps &_rating_maps;
  ParallelRatingMap<NodeID, EdgeWeight> &_parallel_ratings;
  tbb::concurrent_vector<NodeID> &_deferred_nodes;
  CacheAlignedVector<ClusteringSelector::RatedSelection> &_local_selections;
  RoundStatistics &_statistics;
  ClusteringSelector _selector;
};

} // namespace kaminpar::shm::lp
