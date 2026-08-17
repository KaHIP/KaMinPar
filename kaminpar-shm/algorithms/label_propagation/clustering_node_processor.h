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

#include <tbb/concurrent_vector.h>

#include "kaminpar-shm/algorithms/label_propagation/clustering_selector.h"
#include "kaminpar-shm/algorithms/label_propagation/neighborhood_ratings.h"
#include "kaminpar-shm/algorithms/label_propagation/node_processing.h"
#include "kaminpar-shm/algorithms/label_propagation/parallel_rating_map.h"
#include "kaminpar-shm/algorithms/label_propagation/rating_map_pool.h"

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
    const NodeID degree = _graph.degree(u);
    const NodeWeight u_weight = _graph.node_weight(u);
    const NodeID from = _state_view.cluster(u);
    const NodeWeight from_weight = _state_view.cluster_weight(from);
    const bool store_favored_cluster =
        u_weight == from_weight && from_weight <= _state_view.max_cluster_weight(from) / 2;
    const std::size_t upper_bound = std::min<NodeID>(degree, _graph.n());
    const ClusteringSelector::Selection selection =
        local.ratings.execute(upper_bound, [&](auto &ratings) {
          NeighborhoodRatings::accumulate(
              _graph,
              u,
              ratings,
              [&](const NodeID v) { return _state_view.cluster(v); },
              [](const NodeID) { return true; }
          );
          _state_view.deactivate(u);
          const auto result = _selector.select(
              from,
              u_weight,
              store_favored_cluster,
              ratings,
              local.random,
              local.ties,
              local.favored_ties
          );
          ratings.clear();
          return result;
        });

    remember_favored_cluster(u, from, selection, store_favored_cluster);
    record_move(_state_view.commit(_graph, u, from, selection.cluster, u_weight), local);
  }

  void visit_first_phase(const NodeID u, LocalWorker &local) {
    const NodeID degree = _graph.degree(u);
    const NodeWeight u_weight = _graph.node_weight(u);
    const NodeID from = _state_view.cluster(u);
    const NodeWeight from_weight = _state_view.cluster_weight(from);
    const bool store_favored_cluster =
        u_weight == from_weight && from_weight <= _state_view.max_cluster_weight(from) / 2;
    const std::size_t upper_bound = std::min<std::size_t>(
        {degree, _graph.n(), ParallelRatingMap<NodeID, EdgeWeight>::kFlushThreshold}
    );

    const auto selection = local.ratings.execute(upper_bound, [&](auto &ratings) {
      if (NeighborhoodRatings::accumulate_with_capacity(
              _graph,
              u,
              ratings,
              upper_bound,
              ParallelRatingMap<NodeID, EdgeWeight>::kFlushThreshold,
              [&](const NodeID v) { return _state_view.cluster(v); },
              [](const NodeID) { return true; }
          )) [[unlikely]] {
        ratings.clear();
        _deferred_nodes.push_back(u);
        return std::optional<ClusteringSelector::Selection>{};
      }

      _state_view.deactivate(u);
      auto result = _selector.select(
          from,
          u_weight,
          store_favored_cluster,
          ratings,
          local.random,
          local.ties,
          local.favored_ties
      );
      ratings.clear();
      return std::optional<ClusteringSelector::Selection>{result};
    });

    if (selection) {
      remember_favored_cluster(u, from, *selection, store_favored_cluster);
      record_move(_state_view.commit(_graph, u, from, selection->cluster, u_weight), local);
    }
  }

  void finish_work_unit(LocalWorker &local) {
    _state.remove_empty_clusters(local.removed_clusters);
    local.removed_clusters = 0;
  }

  void process_deferred_nodes() {
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
      _state_view.deactivate(u);

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
          _state_view.commit<true>(_graph, u, from, selection.cluster, u_weight);
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
