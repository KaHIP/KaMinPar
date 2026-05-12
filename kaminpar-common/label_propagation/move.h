/*******************************************************************************
 * Label propagation move application.
 *
 * @file:   move.h
 ******************************************************************************/
#pragma once

#include <utility>

#include "kaminpar-common/label_propagation/types.h"

namespace kaminpar::lp {

template <
    typename NodeID,
    typename NodeWeight,
    typename ClusterID,
    typename EdgeWeight,
    typename LabelStore,
    typename WeightStore,
    typename ActiveSetView>
class MoveApplier {
public:
  using Move = NodeMove<NodeID, NodeWeight, ClusterID, EdgeWeight>;
  using Stats = PassStats<NodeID, ClusterID, EdgeWeight>;

  MoveApplier(
      LabelStore &labels,
      WeightStore &weights,
      ActiveSetView &active_set,
      const StopConfig<ClusterID> &config
  )
      : _labels(labels),
        _weights(weights),
        _active_set(active_set),
        _config(config) {}

  KAMINPAR_INLINE std::pair<bool, bool> try_commit(const Move &move, Stats &stats) {
    if (!move.valid || _labels.cluster(move.node) == move.new_cluster) {
      return {false, false};
    }

    const bool successful_weight_move = _weights.move_cluster_weight(
        move.old_cluster,
        move.new_cluster,
        move.node_weight,
        _weights.max_cluster_weight(move.new_cluster)
    );

    if (!successful_weight_move) {
      return {false, false};
    }

    _labels.move_node(move.node, move.new_cluster);
    _active_set.activate_neighbors(move.node);
#ifdef KAMINPAR_ENABLE_STATISTICS
    stats.expected_total_gain += move.gain;
#endif

    const bool emptied_cluster =
        _config.track_cluster_count && _weights.cluster_weight(move.old_cluster) == 0;
    if (emptied_cluster) {
      ++stats.removed_clusters;
    }
    ++stats.moved_nodes;
    return {true, emptied_cluster};
  }

  template <ActiveSetStrategy ActiveSet>
  KAMINPAR_INLINE std::pair<bool, bool> try_commit(const Move &move, Stats &stats) {
    if (!move.valid || _labels.cluster(move.node) == move.new_cluster) {
      return {false, false};
    }

    const bool successful_weight_move = _weights.move_cluster_weight(
        move.old_cluster,
        move.new_cluster,
        move.node_weight,
        _weights.max_cluster_weight(move.new_cluster)
    );

    if (!successful_weight_move) {
      return {false, false};
    }

    _labels.move_node(move.node, move.new_cluster);
    _active_set.template activate_neighbors<ActiveSet>(move.node);
#ifdef KAMINPAR_ENABLE_STATISTICS
    stats.expected_total_gain += move.gain;
#endif

    const bool emptied_cluster =
        _config.track_cluster_count && _weights.cluster_weight(move.old_cluster) == 0;
    if (emptied_cluster) {
      ++stats.removed_clusters;
    }
    ++stats.moved_nodes;
    return {true, emptied_cluster};
  }

  template <ActiveSetStrategy ActiveSet>
  KAMINPAR_INLINE std::pair<bool, bool> try_commit(
      const NodeID node,
      const NodeWeight node_weight,
      const ClusterID old_cluster,
      const ClusterID new_cluster,
      const EdgeWeight gain,
      Stats &stats
  ) {
    if (_labels.cluster(node) == new_cluster) {
      return {false, false};
    }

    const bool successful_weight_move = _weights.move_cluster_weight(
        old_cluster, new_cluster, node_weight, _weights.max_cluster_weight(new_cluster)
    );

    if (!successful_weight_move) {
      return {false, false};
    }

    _labels.move_node(node, new_cluster);
    _active_set.template activate_neighbors<ActiveSet>(node);
#ifdef KAMINPAR_ENABLE_STATISTICS
    stats.expected_total_gain += gain;
#else
    (void)gain;
#endif

    const bool emptied_cluster =
        _config.track_cluster_count && _weights.cluster_weight(old_cluster) == 0;
    if (emptied_cluster) {
      ++stats.removed_clusters;
    }
    ++stats.moved_nodes;
    return {true, emptied_cluster};
  }

private:
  LabelStore &_labels;
  WeightStore &_weights;
  ActiveSetView &_active_set;
  const StopConfig<ClusterID> &_config;
};

} // namespace kaminpar::lp
