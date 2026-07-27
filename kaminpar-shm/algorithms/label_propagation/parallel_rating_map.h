/*******************************************************************************
 * Concurrent rating accumulation for a single high-degree node.
 *
 * @file:   parallel_rating_map.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <cstddef>
#include <limits>
#include <utility>

#include <tbb/parallel_for.h>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/concurrent_fast_reset_array.h"

namespace kaminpar::shm::lp {

template <typename Value, typename Key> class ParallelRatingMap {
public:
  static constexpr std::size_t kFlushThreshold = 10000;

  void ensure_capacity(const std::size_t capacity) {
    if (_ratings.capacity() < capacity) {
      _ratings.resize(capacity);
    }
  }

  template <typename Graph, typename RatingMapPool, typename GetKey, typename AcceptNeighbor>
  void accumulate(
      const Graph &graph,
      const NodeID u,
      RatingMapPool &local_maps,
      GetKey &&get_key,
      AcceptNeighbor &&accept_neighbor
  ) {
    const auto flush = [&](auto &used_entries, auto &local_map) {
      for (const auto [key, rating] : local_map.entries()) {
        const Value previous = __atomic_fetch_add(&_ratings[key], rating, __ATOMIC_RELAXED);
        if (previous == Value()) {
          used_entries.push_back(key);
        }
      }
      local_map.clear();
    };

    graph.pfor_adjacent_nodes(
        u, std::numeric_limits<NodeID>::max(), 2000, [&](auto &&pfor_adjacent_nodes) {
          auto &used_entries = _ratings.local_used_entries();
          auto &local_map = local_maps.maps().local().small_map();

          pfor_adjacent_nodes([&](const NodeID v, const EdgeWeight weight) {
            if (accept_neighbor(v)) {
              local_map[get_key(v)] += weight;
              if (local_map.size() >= kFlushThreshold) [[unlikely]] {
                flush(used_entries, local_map);
              }
            }
          });
        }
    );

    tbb::parallel_for(local_maps.maps().range(), [&](auto &maps) {
      auto &used_entries = _ratings.local_used_entries();
      for (auto &map : maps) {
        flush(used_entries, map.small_map());
      }
    });
  }

  template <typename Lambda> void for_each_partition_and_reset(Lambda &&lambda) {
    _ratings.iterate_and_reset(std::forward<Lambda>(lambda));
  }

  [[nodiscard]] Value &operator[](const Key key) {
    return _ratings[key];
  }

  void free() {
    _ratings.free();
  }

private:
  ConcurrentFastResetArray<Value, Key> _ratings;
};

} // namespace kaminpar::shm::lp
