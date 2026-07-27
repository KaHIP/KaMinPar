/*******************************************************************************
 * Neighborhood rating accumulation.
 *
 * @file:   neighborhood_ratings.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <cstddef>
#include <utility>

#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm::lp {

struct NeighborhoodRatings {
  template <typename Graph, typename Map, typename GetCluster, typename AcceptNeighbor>
  static void accumulate(
      const Graph &graph,
      const NodeID u,
      Map &map,
      GetCluster &&get_cluster,
      AcceptNeighbor &&accept_neighbor
  ) {
    if constexpr (requires { map.accumulator(); }) {
      auto accumulator = map.accumulator();
      graph.adjacent_nodes(
          u,
          [accumulator,
           &get_cluster,
           &accept_neighbor](const NodeID v, const EdgeWeight weight) mutable {
            if (accept_neighbor(v)) {
              accumulator.add(get_cluster(v), weight);
            }
          }
      );
    } else {
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
        if (accept_neighbor(v)) {
          map[get_cluster(v)] += weight;
        }
      });
    }
  }

  template <typename Graph, typename Map, typename GetCluster, typename AcceptNeighbor>
  [[nodiscard]] static bool accumulate_with_capacity(
      const Graph &graph,
      const NodeID u,
      Map &map,
      const std::size_t upper_bound,
      const std::size_t capacity,
      GetCluster &&get_cluster,
      AcceptNeighbor &&accept_neighbor
  ) {
    if (upper_bound < capacity) {
      accumulate(
          graph,
          u,
          map,
          std::forward<GetCluster>(get_cluster),
          std::forward<AcceptNeighbor>(accept_neighbor)
      );
      return false;
    }

    return accumulate_until_full(
        graph,
        u,
        map,
        capacity,
        std::forward<GetCluster>(get_cluster),
        std::forward<AcceptNeighbor>(accept_neighbor)
    );
  }

  template <typename Graph, typename Map, typename GetCluster, typename AcceptNeighbor>
  [[nodiscard]] static bool accumulate_until_full(
      const Graph &graph,
      const NodeID u,
      Map &map,
      const std::size_t capacity,
      GetCluster &&get_cluster,
      AcceptNeighbor &&accept_neighbor
  ) {
    bool full = false;
    if constexpr (requires { map.accumulator(); }) {
      auto accumulator = map.accumulator();
      graph.adjacent_nodes(
          u,
          [accumulator, capacity, &full, &get_cluster, &accept_neighbor](
              const NodeID v, const EdgeWeight weight
          ) mutable {
            if (accept_neighbor(v)) {
              accumulator.add(get_cluster(v), weight);
              if (accumulator.size() >= capacity) [[unlikely]] {
                full = true;
                return true;
              }
            }
            return false;
          }
      );
    } else {
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
        if (accept_neighbor(v)) {
          map[get_cluster(v)] += weight;
          if (map.size() >= capacity) [[unlikely]] {
            full = true;
            return true;
          }
        }
        return false;
      });
    }
    return full;
  }
};

} // namespace kaminpar::shm::lp
