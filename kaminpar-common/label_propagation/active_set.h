/*******************************************************************************
 * Active-set handling for label propagation passes.
 *
 * @file:   active_set.h
 ******************************************************************************/
#pragma once

#include "kaminpar-common/assert.h"
#include "kaminpar-common/label_propagation/types.h"

namespace kaminpar::lp {

template <typename NodeID, typename Graph, typename NeighborPolicy, typename Workspace>
class ActiveSetView {
public:
  ActiveSetView(
      const Graph &graph,
      NeighborPolicy &neighbors,
      Workspace &workspace,
      const ActiveSetConfig &config
  )
      : _graph(graph),
        _neighbors(neighbors),
        _workspace(workspace),
        _config(config) {}

  KAMINPAR_INLINE void initialize_node(const NodeID u) {
    if (_config.strategy != ActiveSetStrategy::NONE) {
      _workspace.active_set.flags[u] = 1;
    }
  }

  [[nodiscard]] KAMINPAR_INLINE bool is_active(const NodeID u) const {
    return _config.strategy == ActiveSetStrategy::NONE ||
           (u < _workspace.active_set.flags.size() &&
            __atomic_load_n(&_workspace.active_set.flags[u], __ATOMIC_RELAXED));
  }

  template <ActiveSetStrategy Strategy>
  [[nodiscard]] KAMINPAR_INLINE bool is_active(const NodeID u) const {
    if constexpr (Strategy == ActiveSetStrategy::NONE) {
      return true;
    } else {
      KASSERT(u < _workspace.active_set.flags.size());
      return __atomic_load_n(&_workspace.active_set.flags[u], __ATOMIC_RELAXED);
    }
  }

  KAMINPAR_INLINE void clear(const NodeID u, const bool is_interface_node) {
    if (_config.strategy == ActiveSetStrategy::GLOBAL) {
      __atomic_store_n(&_workspace.active_set.flags[u], 0, __ATOMIC_RELAXED);
    } else if (_config.strategy == ActiveSetStrategy::LOCAL && !is_interface_node) {
      __atomic_store_n(&_workspace.active_set.flags[u], 0, __ATOMIC_RELAXED);
    }
  }

  template <ActiveSetStrategy Strategy>
  KAMINPAR_INLINE void clear(const NodeID u, const bool is_interface_node) {
    if constexpr (Strategy == ActiveSetStrategy::GLOBAL) {
      __atomic_store_n(&_workspace.active_set.flags[u], 0, __ATOMIC_RELAXED);
    } else if constexpr (Strategy == ActiveSetStrategy::LOCAL) {
      if (!is_interface_node) {
        __atomic_store_n(&_workspace.active_set.flags[u], 0, __ATOMIC_RELAXED);
      }
    }
  }

  KAMINPAR_INLINE void activate_neighbors(const NodeID u) {
    if (_config.strategy == ActiveSetStrategy::NONE) {
      return;
    }

    _graph.adjacent_nodes(u, [&](const NodeID v) {
      if (_neighbors.activate(v) && v < _workspace.active_set.flags.size()) {
        __atomic_store_n(&_workspace.active_set.flags[v], 1, __ATOMIC_RELAXED);
      }
    });
  }

  template <ActiveSetStrategy Strategy> KAMINPAR_INLINE void activate_neighbors(const NodeID u) {
    if constexpr (Strategy != ActiveSetStrategy::NONE) {
      _graph.adjacent_nodes(u, [&](const NodeID v) {
        if constexpr (!ActivatesAllNeighbors<NeighborPolicy>::value) {
          if (!_neighbors.activate(v)) {
            return;
          }
        }

        if constexpr (requires { _graph.is_ghost_node(v); }) {
          if (v < _workspace.active_set.flags.size()) {
            __atomic_store_n(&_workspace.active_set.flags[v], 1, __ATOMIC_RELAXED);
          }
        } else {
          __atomic_store_n(&_workspace.active_set.flags[v], 1, __ATOMIC_RELAXED);
        }
      });
    }
  }

  KAMINPAR_INLINE void activate_neighbors_of_ghost_node(const NodeID u) {
    KASSERT(_graph.is_ghost_node(u));
    if (_config.strategy != ActiveSetStrategy::GLOBAL) {
      return;
    }

    _graph.ghost_graph().adjacent_nodes(u, [&](const NodeID v) {
      if (_neighbors.activate(v) && v < _workspace.active_set.flags.size()) {
        __atomic_store_n(&_workspace.active_set.flags[v], 1, __ATOMIC_RELAXED);
      }
    });
  }

private:
  const Graph &_graph;
  NeighborPolicy &_neighbors;
  Workspace &_workspace;
  const ActiveSetConfig &_config;
};

} // namespace kaminpar::lp
