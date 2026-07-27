/*******************************************************************************
 * Neighborhood rating accumulation.
 *
 * @file:   neighborhood_ratings.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <cstddef>

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
    graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight weight) {
      if (accept_neighbor(v)) {
        map[get_cluster(v)] += weight;
      }
    });
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
    return full;
  }
};

} // namespace kaminpar::shm::lp
