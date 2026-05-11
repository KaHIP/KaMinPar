/*******************************************************************************
 * Shared label propagation primitives.
 *
 * @file:   kernel.h
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <limits>
#include <type_traits>
#include <utility>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_invoke.h>

#include "kaminpar-common/assert.h"
#include "kaminpar-common/label_propagation/active_set.h"
#include "kaminpar-common/label_propagation/move.h"
#include "kaminpar-common/label_propagation/rating_accumulator.h"
#include "kaminpar-common/label_propagation/types.h"
#include "kaminpar-common/parallel/atomic.h"
#include "kaminpar-common/random.h"

namespace kaminpar::lp {

template <
    typename Graph,
    typename LabelStore,
    typename WeightStore,
    typename ClusterSelector,
    typename NeighborPolicy,
    typename Workspace>
class LabelPropagationKernel {
public:
  using NodeID = typename Graph::NodeID;
  using NodeWeight = typename Graph::NodeWeight;
  using EdgeID = typename Graph::EdgeID;
  using EdgeWeight = typename Graph::EdgeWeight;
  using ClusterID = typename LabelStore::ClusterIDType;
  using ClusterWeight = typename WeightStore::ClusterWeightType;
  using RatingMap = typename Workspace::RatingMapType;
  using GrowingRatingMap = typename Workspace::GrowingRatingMapType;
  using ConcurrentRatingMap = typename Workspace::ConcurrentRatingMapType;
  using WorkspaceType = Workspace;
  using SelectionContext = NodeContext<NodeID, NodeWeight, ClusterID, ClusterWeight, EdgeWeight>;
  using Result = PassResult<NodeID, ClusterID, EdgeWeight>;
  using ActiveSet = ActiveSetView<NodeID, Graph, NeighborPolicy, Workspace>;
  using Move = NodeMove<NodeID, NodeWeight, ClusterID, EdgeWeight>;
  using Stats = PassStats<NodeID, ClusterID, EdgeWeight>;
  using MoveApplier = lp::
      MoveApplier<NodeID, NodeWeight, ClusterID, EdgeWeight, LabelStore, WeightStore, ActiveSet>;
  using RatingAccumulator = lp::RatingAccumulator<NodeID, Graph, LabelStore, NeighborPolicy>;

  LabelPropagationKernel(
      const Graph &graph,
      LabelStore &labels,
      WeightStore &weights,
      ClusterSelector &selector,
      NeighborPolicy &neighbors,
      Workspace &workspace,
      PassConfig<NodeID, ClusterID> config
  )
      : _graph(graph),
        _labels(labels),
        _weights(weights),
        _selector(selector),
        _neighbors(neighbors),
        _workspace(workspace),
        _config(config),
        _active_set(graph, neighbors, workspace, _config.active_set),
        _move_applier(labels, weights, _active_set, _config.stopping),
        _rating_accumulator(graph, labels, neighbors, _config.nodes, _config.active_set) {}

  void set_config(const PassConfig<NodeID, ClusterID> config) {
    _config = config;
  }

  [[nodiscard]] const PassConfig<NodeID, ClusterID> &config() const {
    return _config;
  }

  void initialize(const Initialization<NodeID, ClusterID> init) {
    _num_nodes = init.num_nodes;
    _num_active_nodes = init.num_active_nodes;
    _prev_num_clusters = _num_clusters;
    _num_clusters = init.num_clusters;
    _initial_num_clusters = init.num_clusters;
    _current_num_clusters = init.num_clusters;
    _relabeled = false;
    _workspace.allocate(_num_nodes, _num_active_nodes, _num_clusters, _prev_num_clusters, _config);
    reset_state();
  }

  void clear_iteration_order_cache() {}

  [[nodiscard]] const Graph &graph() const {
    return _graph;
  }

  [[nodiscard]] LabelStore &labels() {
    return _labels;
  }

  [[nodiscard]] const LabelStore &labels() const {
    return _labels;
  }

  [[nodiscard]] WeightStore &weights() {
    return _weights;
  }

  [[nodiscard]] const WeightStore &weights() const {
    return _weights;
  }

  [[nodiscard]] ClusterSelector &selector() {
    return _selector;
  }

  [[nodiscard]] NeighborPolicy &neighbors() {
    return _neighbors;
  }

  [[nodiscard]] Workspace &workspace() {
    return _workspace;
  }

  [[nodiscard]] const Workspace &workspace() const {
    return _workspace;
  }

  [[nodiscard]] NodeID num_nodes() const {
    return _num_nodes;
  }

  [[nodiscard]] NodeID num_active_nodes() const {
    return _num_active_nodes;
  }

  [[nodiscard]] ClusterID initial_num_clusters() const {
    return _initial_num_clusters;
  }

  [[nodiscard]] ClusterID current_num_clusters() const {
    return _current_num_clusters;
  }

  [[nodiscard]] bool relabeled() const {
    return _relabeled;
  }

  void set_initial_num_clusters(const ClusterID num_clusters) {
    _initial_num_clusters = num_clusters;
  }

  void set_relabeled(const bool relabeled) {
    _relabeled = relabeled;
  }

  void decrement_current_num_clusters() {
    --_current_num_clusters;
  }

  [[nodiscard]] bool should_stop() const {
    return _config.stopping.track_cluster_count &&
           _current_num_clusters <= _config.stopping.desired_clusters;
  }

  void reset_state() {
    tbb::parallel_invoke(
        [&] {
          tbb::parallel_for<NodeID>(0, _num_active_nodes, [&](const NodeID u) {
            _active_set.initialize_node(u);

            const ClusterID initial_cluster = _labels.initial_cluster(u);
            _labels.init_cluster(u, initial_cluster);
            if (_config.selection.track_favored_clusters &&
                u < _workspace.postprocessing.favored_clusters.size()) {
              _workspace.postprocessing.favored_clusters[u] = initial_cluster;
            }
            if (u < _workspace.postprocessing.moved.size()) {
              _workspace.postprocessing.moved[u] = 0;
            }
            if constexpr (requires(LabelStore labels, NodeID node) {
                            labels.reset_node_state(node);
                          }) {
              _labels.reset_node_state(u);
            }
          });
        },
        [&] {
          tbb::parallel_for<ClusterID>(0, _initial_num_clusters, [&](const ClusterID cluster) {
            _weights.init_cluster_weight(cluster, _weights.initial_cluster_weight(cluster));
          });
        }
    );
    _expected_total_gain = 0;
    _current_num_clusters = _initial_num_clusters;
  }

  [[nodiscard]] KAMINPAR_INLINE bool should_consider(const NodeID u) const {
    if (u >= _num_active_nodes) {
      return false;
    }
    if (_neighbors.skip(u)) {
      return false;
    }
    if (_graph.degree(u) >= _config.nodes.max_degree) {
      return false;
    }
    if (!_active_set.is_active(u)) {
      return false;
    }
    return true;
  }

  template <typename RatingMap>
  KAMINPAR_INLINE void rate_neighbors(const NodeID u, RatingMap &map, bool &is_interface_node) {
    _rating_accumulator.rate_neighbors(u, map, _num_active_nodes, is_interface_node);
  }

  template <typename RatingMap>
  [[nodiscard]] KAMINPAR_INLINE bool rate_neighbors_until(
      const NodeID u, RatingMap &map, const std::size_t max_map_size, bool &is_interface_node
  ) {
    return _rating_accumulator.rate_neighbors_until(
        u, map, _num_active_nodes, max_map_size, is_interface_node
    );
  }

  KAMINPAR_INLINE void clear_active(const NodeID u, const bool is_interface_node) {
    _active_set.clear(u, is_interface_node);
  }

  template <
      TieBreakingStrategy TieBreaking,
      typename ActualMap,
      typename TieBreakingClusters,
      typename TieBreakingFavoredClusters>
  [[nodiscard]] KAMINPAR_INLINE Move select_move(
      const NodeID u,
      const NodeWeight u_weight,
      const ClusterID u_cluster,
      Random &rand,
      ActualMap &map,
      TieBreakingClusters &tie_breaking_clusters,
      TieBreakingFavoredClusters &tie_breaking_favored_clusters
  ) {
    const ClusterWeight initial_cluster_weight = _weights.cluster_weight(u_cluster);
    const bool track_favored_cluster =
        _config.selection.track_favored_clusters && u_weight == initial_cluster_weight &&
        initial_cluster_weight <= _weights.max_cluster_weight(u_cluster) / 2;
    const EdgeWeight gain_delta = _config.selection.use_actual_gain ? map[u_cluster] : 0;
    SelectionContext context{
        .rand = rand,
        .node = u,
        .node_weight = u_weight,
        .initial_cluster = u_cluster,
        .initial_cluster_weight = initial_cluster_weight,
        .gain_delta = gain_delta,
        .track_favored_cluster = track_favored_cluster,
    };

    const auto choice = _selector.template select<TieBreaking>(
        context, map, tie_breaking_clusters, tie_breaking_favored_clusters
    );

    if (track_favored_cluster && choice.best_cluster == context.initial_cluster) {
      _workspace.postprocessing.favored_clusters[u] = choice.favored_cluster;
    }

    const EdgeWeight actual_gain = choice.best_gain - map[context.initial_cluster];
    map.clear();
    return {
        .node = u,
        .node_weight = u_weight,
        .old_cluster = u_cluster,
        .new_cluster = choice.best_cluster,
        .gain = actual_gain,
        .valid = true,
    };
  }

  template <
      TieBreakingStrategy TieBreaking,
      typename LocalRatingMap,
      typename TieBreakingClusters,
      typename TieBreakingFavoredClusters>
  [[nodiscard]] KAMINPAR_INLINE Move find_best_move(
      const NodeID u,
      Random &rand,
      LocalRatingMap &rating_map,
      TieBreakingClusters &tie_breaking_clusters,
      TieBreakingFavoredClusters &tie_breaking_favored_clusters
  ) {
    const NodeWeight u_weight = _graph.node_weight(u);
    const ClusterID u_cluster = _labels.cluster(u);

    const auto action = [&](auto &map) {
      bool is_interface_node = false;
      rate_neighbors(u, map, is_interface_node);
      clear_active(u, is_interface_node);
      return select_move<TieBreaking>(
          u, u_weight, u_cluster, rand, map, tie_breaking_clusters, tie_breaking_favored_clusters
      );
    };

    if constexpr (std::is_same_v<LocalRatingMap, GrowingRatingMap>) {
      return action(rating_map);
    } else {
      const std::size_t upper_bound_size =
          std::min<ClusterID>(_graph.degree(u), _initial_num_clusters);
      return rating_map.execute(upper_bound_size, action);
    }
  }

  [[nodiscard]] KAMINPAR_INLINE Move find_best_move(const NodeID u, Random &rand) {
    auto &rating_map = _workspace.rating.maps.local();
    auto &tie_breaking_clusters = _workspace.selection.tie_breaking_clusters.local();
    auto &tie_breaking_favored_clusters =
        _workspace.selection.tie_breaking_favored_clusters.local();

    switch (_config.selection.tie_breaking_strategy) {
    case TieBreakingStrategy::GEOMETRIC:
      return find_best_move<TieBreakingStrategy::GEOMETRIC>(
          u, rand, rating_map, tie_breaking_clusters, tie_breaking_favored_clusters
      );
    case TieBreakingStrategy::UNIFORM:
      return find_best_move<TieBreakingStrategy::UNIFORM>(
          u, rand, rating_map, tie_breaking_clusters, tie_breaking_favored_clusters
      );
    }
  }

  KAMINPAR_INLINE std::pair<bool, bool> commit(const Move &move, Stats &stats) {
    return _move_applier.try_commit(move, stats);
  }

  KAMINPAR_INLINE std::pair<bool, bool> try_commit_move(const Move &move, Stats &stats) {
    return commit(move, stats);
  }

  KAMINPAR_INLINE void activate_neighbors_of_ghost_node(const NodeID u) {
    _active_set.activate_neighbors_of_ghost_node(u);
  }

  [[nodiscard]] Result finish_pass(const tbb::enumerable_thread_specific<Stats> &stats_ets) {
    Result result;
    for (const Stats &local_stats : stats_ets) {
      result.processed_nodes += local_stats.processed_nodes;
      result.moved_nodes += local_stats.moved_nodes;
      result.removed_clusters += local_stats.removed_clusters;
      result.expected_total_gain += local_stats.expected_total_gain;
    }

    _current_num_clusters -= result.removed_clusters;
    _expected_total_gain += result.expected_total_gain;
    return result;
  }

private:
  const Graph &_graph;
  LabelStore &_labels;
  WeightStore &_weights;
  ClusterSelector &_selector;
  NeighborPolicy &_neighbors;
  Workspace &_workspace;
  PassConfig<NodeID, ClusterID> _config;
  ActiveSet _active_set;
  MoveApplier _move_applier;
  RatingAccumulator _rating_accumulator;

  NodeID _num_nodes = 0;
  NodeID _num_active_nodes = 0;
  ClusterID _num_clusters = 0;
  ClusterID _prev_num_clusters = 0;
  ClusterID _initial_num_clusters = 0;
  parallel::Atomic<ClusterID> _current_num_clusters = 0;
  parallel::Atomic<EdgeWeight> _expected_total_gain = 0;
  bool _relabeled = false;
};

} // namespace kaminpar::lp
