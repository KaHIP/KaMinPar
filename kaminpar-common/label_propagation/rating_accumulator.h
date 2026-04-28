/*******************************************************************************
 * Neighbor rating accumulation for label propagation.
 *
 * @file:   rating_accumulator.h
 ******************************************************************************/
#pragma once

#include <limits>

#include "kaminpar-common/label_propagation/types.h"

namespace kaminpar::lp {

template <typename NodeID, typename Graph, typename LabelStore, typename NeighborPolicy>
class RatingAccumulator {
public:
  using EdgeWeight = typename Graph::EdgeWeight;

  RatingAccumulator(
      const Graph &graph,
      LabelStore &labels,
      NeighborPolicy &neighbors,
      const NodeLimits<NodeID> &node_limits,
      const ActiveSetConfig &active_set_config
  )
      : _graph(graph),
        _labels(labels),
        _neighbors(neighbors),
        _node_limits(node_limits),
        _active_set_config(active_set_config) {}

  template <typename RatingMap>
  KAMINPAR_LP_INLINE void rate_neighbors(
      const NodeID u, RatingMap &map, const NodeID num_active_nodes, bool &is_interface_node
  ) {
    const auto add_to_rating_map = [&](const NodeID v, const EdgeWeight w) {
      if (_neighbors.accept(u, v)) {
        const auto v_cluster = _labels.cluster(v);
        map[v_cluster] += w;

        if (_active_set_config.strategy == ActiveSetStrategy::LOCAL) {
          is_interface_node |= v >= num_active_nodes;
        }
      }
    };

    if (_node_limits.max_neighbors == std::numeric_limits<NodeID>::max()) [[likely]] {
      _graph.adjacent_nodes(u, add_to_rating_map);
    } else {
      _graph.adjacent_nodes(u, _node_limits.max_neighbors, add_to_rating_map);
    }
  }

  template <typename RatingMap>
  [[nodiscard]] KAMINPAR_LP_INLINE bool rate_neighbors_until(
      const NodeID u,
      RatingMap &map,
      const NodeID num_active_nodes,
      const std::size_t max_map_size,
      bool &is_interface_node
  ) {
    bool reached_limit = false;
    const auto add_to_rating_map = [&](const NodeID v, const EdgeWeight w) -> bool {
      if (_neighbors.accept(u, v)) {
        const auto v_cluster = _labels.cluster(v);
        map[v_cluster] += w;

        if (map.size() >= max_map_size) [[unlikely]] {
          reached_limit = true;
          return true;
        }

        if (_active_set_config.strategy == ActiveSetStrategy::LOCAL) {
          is_interface_node |= v >= num_active_nodes;
        }
      }

      return false;
    };

    if (_node_limits.max_neighbors == std::numeric_limits<NodeID>::max()) [[likely]] {
      _graph.adjacent_nodes(u, add_to_rating_map);
    } else {
      _graph.adjacent_nodes(u, _node_limits.max_neighbors, add_to_rating_map);
    }

    return reached_limit;
  }

private:
  const Graph &_graph;
  LabelStore &_labels;
  NeighborPolicy &_neighbors;
  const NodeLimits<NodeID> &_node_limits;
  const ActiveSetConfig &_active_set_config;
};

} // namespace kaminpar::lp
