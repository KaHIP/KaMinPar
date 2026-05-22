/*******************************************************************************
 * @file:   node_tracker.h
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 ******************************************************************************/
#pragma once

#include <tbb/parallel_for.h>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::fm {

class NodeTracker {
public:
  static constexpr int UNLOCKED = 0;
  static constexpr int MOVED_LOCALLY = -1;
  static constexpr int MOVED_GLOBALLY = -2;

  NodeTracker(const NodeID max_n) : _state(max_n) {}

  bool lock(const NodeID u, const int id) {
    int free = 0;
    return __atomic_compare_exchange_n(
        &_state[u], &free, id, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
    );
  }

  [[nodiscard]] int owner(const NodeID u) const {
    return __atomic_load_n(&_state[u], __ATOMIC_RELAXED);
  }

  void set(const NodeID node, const int value) {
    __atomic_store_n(&_state[node], value, __ATOMIC_RELAXED);
  }

  void reset() {
    tbb::parallel_for<NodeID>(0, _state.size(), [&](const NodeID node) {
      _state[node] = UNLOCKED;
    });
  }

  void free() {
    _state.free();
  }

private:
  StaticArray<int> _state;
};

} // namespace kaminpar::shm::fm
