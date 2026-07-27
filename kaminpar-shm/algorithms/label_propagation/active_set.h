/*******************************************************************************
 * Parallel active set used by iterative graph algorithms.
 *
 * @file:   active_set.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <limits>

#include <tbb/parallel_for.h>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/inline.h"

namespace kaminpar::shm::lp {

class ActiveSetView {
public:
  explicit ActiveSetView(std::uint8_t *active) : _active(active) {}

  [[nodiscard]] KAMINPAR_INLINE bool contains(const NodeID u) const {
    return __atomic_load_n(&_active[u], __ATOMIC_RELAXED);
  }

  KAMINPAR_INLINE void deactivate(const NodeID u) {
    __atomic_store_n(&_active[u], static_cast<std::uint8_t>(0), __ATOMIC_RELAXED);
  }

  KAMINPAR_INLINE void activate(const NodeID u) {
    __atomic_store_n(&_active[u], static_cast<std::uint8_t>(1), __ATOMIC_RELAXED);
  }

  template <typename Graph, typename AcceptNeighbor>
  KAMINPAR_INLINE void
  activate_neighbors(const Graph &graph, const NodeID u, AcceptNeighbor &&accept_neighbor) {
    graph.adjacent_nodes(u, [&](const NodeID v) {
      if (accept_neighbor(v)) {
        activate(v);
      }
    });
  }

  template <typename Graph, typename AcceptNeighbor>
  void activate_neighbors_parallel(
      const Graph &graph, const NodeID u, AcceptNeighbor &&accept_neighbor
  ) {
    graph.pfor_adjacent_nodes(
        u, std::numeric_limits<NodeID>::max(), 20000, [&](const NodeID v, const EdgeWeight) {
          if (accept_neighbor(v)) {
            activate(v);
          }
        }
    );
  }

private:
  std::uint8_t *_active;
};

class ActiveSet {
public:
  void resize(const NodeID num_nodes) {
    if (_active.size() < num_nodes) {
      _active.resize(num_nodes);
    }
  }

  void reset(const NodeID num_nodes) {
    resize(num_nodes);
    tbb::parallel_for<NodeID>(0, num_nodes, [&](const NodeID u) {
      _active[u] = static_cast<std::uint8_t>(1);
    });
  }

  [[nodiscard]] bool contains(const NodeID u) const {
    return __atomic_load_n(&_active[u], __ATOMIC_RELAXED);
  }

  void deactivate(const NodeID u) {
    __atomic_store_n(&_active[u], static_cast<std::uint8_t>(0), __ATOMIC_RELAXED);
  }

  void initialize(const NodeID u) {
    _active[u] = static_cast<std::uint8_t>(1);
  }

  void activate(const NodeID u) {
    __atomic_store_n(&_active[u], static_cast<std::uint8_t>(1), __ATOMIC_RELAXED);
  }

  [[nodiscard]] ActiveSetView view() {
    return ActiveSetView(_active.data());
  }

  template <typename Graph, typename AcceptNeighbor>
  void activate_neighbors(const Graph &graph, const NodeID u, AcceptNeighbor &&accept_neighbor) {
    graph.adjacent_nodes(u, [&](const NodeID v) {
      if (accept_neighbor(v)) {
        activate(v);
      }
    });
  }

  void free() {
    _active.free();
  }

private:
  StaticArray<std::uint8_t> _active;
};

} // namespace kaminpar::shm::lp
