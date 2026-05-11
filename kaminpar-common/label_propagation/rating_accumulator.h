/*******************************************************************************
 * Neighbor rating accumulation for label propagation.
 *
 * @file:   rating_accumulator.h
 ******************************************************************************/
#pragma once

#include <algorithm>
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
        _active_set_config(active_set_config),
        _unit_edge_weights([&] {
          if constexpr (requires { graph.is_edge_weighted(); }) {
            return !graph.is_edge_weighted();
          } else {
            return false;
          }
        }()) {}

  template <typename RatingMap>
  KAMINPAR_INLINE void rate_neighbors(
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

  template <ActiveSetStrategy ActiveSet, typename RatingMap>
  KAMINPAR_INLINE void rate_neighbors(
      const NodeID u, RatingMap &map, const NodeID num_active_nodes, bool &is_interface_node
  ) {
    if constexpr (requires {
                    _graph.raw_nodes();
                    _graph.raw_edges();
                  }) {
      rate_raw_neighbors<ActiveSet>(u, map, num_active_nodes, is_interface_node);
      return;
    }

    const auto add_to_rating_map = [&](const NodeID v, const EdgeWeight w) {
      if constexpr (!AcceptsAllNeighbors<NeighborPolicy>::value) {
        if (!_neighbors.accept(u, v)) {
          return;
        }
      }

      const auto v_cluster = _labels.cluster(v);
      map[v_cluster] += w;

      if constexpr (ActiveSet == ActiveSetStrategy::LOCAL) {
        is_interface_node |= v >= num_active_nodes;
      }
    };

    if (_node_limits.max_neighbors == std::numeric_limits<NodeID>::max()) [[likely]] {
      _graph.adjacent_nodes(u, add_to_rating_map);
    } else {
      _graph.adjacent_nodes(u, _node_limits.max_neighbors, add_to_rating_map);
    }
  }

  template <typename RatingMap>
  [[nodiscard]] KAMINPAR_INLINE bool rate_neighbors_until(
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

  template <ActiveSetStrategy ActiveSet, typename RatingMap>
  [[nodiscard]] KAMINPAR_INLINE bool rate_neighbors_until(
      const NodeID u,
      RatingMap &map,
      const NodeID num_active_nodes,
      const std::size_t max_map_size,
      bool &is_interface_node
  ) {
    if constexpr (requires {
                    _graph.raw_nodes();
                    _graph.raw_edges();
                  }) {
      return rate_raw_neighbors_until<ActiveSet>(
          u, map, num_active_nodes, max_map_size, is_interface_node
      );
    }

    bool reached_limit = false;
    const auto add_to_rating_map = [&](const NodeID v, const EdgeWeight w) -> bool {
      if constexpr (!AcceptsAllNeighbors<NeighborPolicy>::value) {
        if (!_neighbors.accept(u, v)) {
          return false;
        }
      }

      const auto v_cluster = _labels.cluster(v);
      map[v_cluster] += w;

      if (map.size() >= max_map_size) [[unlikely]] {
        reached_limit = true;
        return true;
      }

      if constexpr (ActiveSet == ActiveSetStrategy::LOCAL) {
        is_interface_node |= v >= num_active_nodes;
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
  template <ActiveSetStrategy ActiveSet, typename RatingMap>
  KAMINPAR_INLINE void rate_raw_neighbors(
      const NodeID u, RatingMap &map, const NodeID num_active_nodes, bool &is_interface_node
  ) {
    constexpr std::size_t kPrefetchDistance = 16;

    const auto &nodes = _graph.raw_nodes();
    const auto &edges = _graph.raw_edges();
    const auto from = nodes[u];
    const NodeID degree = static_cast<NodeID>(nodes[u + 1] - from);
    const auto to = from + std::min(degree, _node_limits.max_neighbors);

    const auto rate_edges = [&](auto &&edge_weight) {
      for (auto edge = from; edge < to; ++edge) {
        if constexpr (requires(NodeID node) { _labels.prefetch_cluster(node); }) {
          if (edge + kPrefetchDistance < to) {
            _labels.prefetch_cluster(edges[edge + kPrefetchDistance]);
          }
        }

        const NodeID v = edges[edge];
        if constexpr (!AcceptsAllNeighbors<NeighborPolicy>::value) {
          if (!_neighbors.accept(u, v)) {
            continue;
          }
        }

        const auto v_cluster = _labels.cluster(v);
        map[v_cluster] += edge_weight(edge);

        if constexpr (ActiveSet == ActiveSetStrategy::LOCAL) {
          is_interface_node |= v >= num_active_nodes;
        }
      }
    };

    if (!_unit_edge_weights) {
      const auto &edge_weights = _graph.raw_edge_weights();
      rate_edges([&](const auto edge) { return edge_weights[edge]; });
    } else {
      rate_edges([](const auto) { return 1; });
    }
  }

  template <ActiveSetStrategy ActiveSet, typename RatingMap>
  [[nodiscard]] KAMINPAR_INLINE bool rate_raw_neighbors_until(
      const NodeID u,
      RatingMap &map,
      const NodeID num_active_nodes,
      const std::size_t max_map_size,
      bool &is_interface_node
  ) {
    constexpr std::size_t kPrefetchDistance = 16;

    const auto &nodes = _graph.raw_nodes();
    const auto &edges = _graph.raw_edges();
    const auto from = nodes[u];
    const NodeID degree = static_cast<NodeID>(nodes[u + 1] - from);
    const auto to = from + std::min(degree, _node_limits.max_neighbors);

    bool reached_limit = false;
    const auto rate_edges = [&](auto &&edge_weight) {
      for (auto edge = from; edge < to; ++edge) {
        if constexpr (requires(NodeID node) { _labels.prefetch_cluster(node); }) {
          if (edge + kPrefetchDistance < to) {
            _labels.prefetch_cluster(edges[edge + kPrefetchDistance]);
          }
        }

        const NodeID v = edges[edge];
        if constexpr (!AcceptsAllNeighbors<NeighborPolicy>::value) {
          if (!_neighbors.accept(u, v)) {
            continue;
          }
        }

        const auto v_cluster = _labels.cluster(v);
        map[v_cluster] += edge_weight(edge);

        if (map.size() >= max_map_size) [[unlikely]] {
          reached_limit = true;
          return;
        }

        if constexpr (ActiveSet == ActiveSetStrategy::LOCAL) {
          is_interface_node |= v >= num_active_nodes;
        }
      }
    };

    if (!_unit_edge_weights) {
      const auto &edge_weights = _graph.raw_edge_weights();
      rate_edges([&](const auto edge) { return edge_weights[edge]; });
    } else {
      rate_edges([](const auto) { return 1; });
    }

    return reached_limit;
  }

  const Graph &_graph;
  LabelStore &_labels;
  NeighborPolicy &_neighbors;
  const NodeLimits<NodeID> &_node_limits;
  const ActiveSetConfig &_active_set_config;
  bool _unit_edge_weights;
};

} // namespace kaminpar::lp
