/*******************************************************************************
 * @file:   ghost_node_mapper.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Utility data structure to construct the ghost node mapping.
 ******************************************************************************/
#pragma once

#include <kassert/kassert.hpp>
#include <tbb/concurrent_hash_map.h>

#include "dkaminpar/datastructures/growt.h"
#include "dkaminpar/definitions.h"
#include "dkaminpar/mpi/wrapper.h"

#include "common/datastructures/static_array.h"

namespace kaminpar::dist::graph {
class GhostNodeMapper {
  // @todo replace by growt hash table
  using GhostNodeMap = tbb::concurrent_hash_map<GlobalNodeID, NodeID>;

public:
  struct Result {
    growt::StaticGhostNodeMapping global_to_ghost;
    StaticArray<GlobalNodeID> ghost_to_global;
    StaticArray<PEID> ghost_owner;
  };

  explicit GhostNodeMapper(
      PEID rank, const StaticArray<GlobalNodeID> &node_distribution
  )
      : _node_distribution(node_distribution.size()),
        _n(static_cast<NodeID>(
            _node_distribution[rank + 1] - _node_distribution[rank]
        )),
        _next_ghost_node(_n) {
    std::copy(
        node_distribution.begin(),
        node_distribution.end(),
        _node_distribution.begin()
    );
  }

  NodeID new_ghost_node(const GlobalNodeID global_node) {
    GhostNodeMap::accessor entry;
    if (_global_to_ghost.insert(entry, global_node)) {
      const NodeID ghost_node =
          _next_ghost_node.fetch_add(1, std::memory_order_relaxed);
      entry->second = ghost_node;
    } else {
      [[maybe_unused]] const bool found =
          _global_to_ghost.find(entry, global_node);
      KASSERT(found);
    }

    return entry->second;
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
        const auto owner_it = std::upper_bound(
            _node_distribution.begin() + 1,
            _node_distribution.end(),
            global_node
        );
        const PEID owner = static_cast<PEID>(
            std::distance(_node_distribution.begin(), owner_it) - 1
        );

        ghost_to_global[local_ghost] = global_node;
        ghost_owner[local_ghost] = owner;
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
