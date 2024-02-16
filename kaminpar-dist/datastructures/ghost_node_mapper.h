/*******************************************************************************
 * Utility data structure to construct the ghost node mapping.
 *
 * @file:   ghost_node_mapper.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#pragma once

#include <tbb/concurrent_hash_map.h>

#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/datastructures/growt.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/logger.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::dist::graph {
class GhostNodeMapper {
  SET_DEBUG(false);

  // @todo replace by growt hash table
  using GhostNodeMap = tbb::concurrent_hash_map<GlobalNodeID, NodeID>;

public:
  struct Result {
    growt::StaticGhostNodeMapping global_to_ghost;
    StaticArray<GlobalNodeID> ghost_to_global;
    StaticArray<PEID> ghost_owner;
  };

  GhostNodeMapper(PEID rank, const StaticArray<GlobalNodeID> &node_distribution)
      : _node_distribution(node_distribution.begin(), node_distribution.end()),
        _n(static_cast<NodeID>(_node_distribution[rank + 1] - _node_distribution[rank])),
        _next_ghost_node(_n) {}

  NodeID new_ghost_node(const GlobalNodeID global_node) {
    GhostNodeMap::accessor entry;
    if (_global_to_ghost.insert(entry, global_node)) {
      const NodeID ghost_node = _next_ghost_node.fetch_add(1, std::memory_order_relaxed);
      entry->second = ghost_node;
    } else {
      [[maybe_unused]] const bool found = _global_to_ghost.find(entry, global_node);
      KASSERT(found);
    }

    DBG << "Mapping " << global_node << " to " << entry->second;
    return entry->second;
  }

  [[nodiscard]] NodeID next_ghost_node() const {
    return _next_ghost_node;
  }

  [[nodiscard]] Result finalize() {
    const NodeID ghost_n = static_cast<NodeID>(_next_ghost_node - _n);

    growt::StaticGhostNodeMapping global_to_ghost(ghost_n);
    StaticArray<GlobalNodeID> ghost_to_global(ghost_n);
    StaticArray<PEID> ghost_owner(ghost_n);

    tbb::parallel_for(_global_to_ghost.range(), [&](const auto r) {
      for (auto it = r.begin(); it != r.end(); ++it) {
        const GlobalNodeID global_node = it->first;
        const NodeID local_node = it->second;
        const NodeID local_ghost = local_node - _n;

        const auto owner_it =
            std::upper_bound(_node_distribution.begin() + 1, _node_distribution.end(), global_node);
        const PEID owner =
            static_cast<PEID>(std::distance(_node_distribution.begin(), owner_it) - 1);

        KASSERT(local_ghost < ghost_to_global.size());
        KASSERT(local_ghost < ghost_owner.size());

        ghost_to_global[local_ghost] = global_node;
        ghost_owner[local_ghost] = owner;

        DBG << "Map global node " << global_node << " to local ghost node " << local_node;
        global_to_ghost.insert(global_node + 1, local_node);
      }
    });

    return {
        .global_to_ghost = std::move(global_to_ghost),
        .ghost_to_global = std::move(ghost_to_global),
        .ghost_owner = std::move(ghost_owner)};
  }

private:
  StaticArray<GlobalNodeID> _node_distribution;
  NodeID _n;
  parallel::Atomic<NodeID> _next_ghost_node;
  GhostNodeMap _global_to_ghost;
};
} // namespace kaminpar::dist::graph
