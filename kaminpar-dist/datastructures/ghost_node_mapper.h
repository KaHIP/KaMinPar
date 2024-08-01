/*******************************************************************************
 * Utility data structure to construct the ghost node mapping.
 *
 * @file:   ghost_node_mapper.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 ******************************************************************************/
#pragma once

#include <mpi.h>
#include <tbb/concurrent_hash_map.h>

#include "kaminpar-mpi/definitions.h"

#include "kaminpar-dist/datastructures/growt.h"
#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/bitvector_rank.h"
#include "kaminpar-common/datastructures/compact_static_array.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/atomic.h"

namespace kaminpar::dist {

class CompactGhostNodeMapping {
public:
  explicit CompactGhostNodeMapping(
      const NodeID num_nodes,
      const NodeID num_ghost_nodes,
      RankCombinedBitVector<> global_to_ghost_bitmap,
      CompactStaticArray<NodeID> dense_global_to_ghost,
      CompactStaticArray<GlobalNodeID> ghost_to_global,
      CompactStaticArray<UPEID> ghost_owner
  )
      : _num_nodes(num_nodes),
        _num_ghost_nodes(num_ghost_nodes),
        _use_dense_global_to_ghost(true),
        _global_to_ghost_bitmap(std::move(global_to_ghost_bitmap)),
        _dense_global_to_ghost(std::move(dense_global_to_ghost)),
        _ghost_to_global(std::move(ghost_to_global)),
        _ghost_owner(std::move(ghost_owner)) {}

  explicit CompactGhostNodeMapping(
      const NodeID num_nodes,
      const NodeID num_ghost_nodes,
      growt::StaticGhostNodeMapping sparse_global_to_ghost,
      CompactStaticArray<GlobalNodeID> ghost_to_global,
      CompactStaticArray<UPEID> ghost_owner
  )
      : _num_nodes(num_nodes),
        _num_ghost_nodes(num_ghost_nodes),
        _use_dense_global_to_ghost(false),
        _sparse_global_to_ghost(std::move(sparse_global_to_ghost)),
        _ghost_to_global(std::move(ghost_to_global)),
        _ghost_owner(std::move(ghost_owner)) {}

  [[nodiscard]] NodeID num_ghost_nodes() const {
    return _num_ghost_nodes;
  }

  [[nodiscard]] bool contains_global_as_ghost(const GlobalNodeID global_node) const {
    if (_use_dense_global_to_ghost) [[likely]] {
      return _global_to_ghost_bitmap.is_set(global_node);
    } else {
      return _sparse_global_to_ghost.find(global_node + 1) != _sparse_global_to_ghost.end();
    }
  }

  [[nodiscard]] NodeID global_to_ghost(const GlobalNodeID global_node) const {
    if (_use_dense_global_to_ghost) [[likely]] {
      const NodeID dense_index = _global_to_ghost_bitmap.rank(global_node);
      return _dense_global_to_ghost[dense_index] + _num_nodes;
    } else {
      return (*_sparse_global_to_ghost.find(global_node + 1)).second;
    }
  }

  [[nodiscard]] GlobalNodeID ghost_to_global(const NodeID ghost_node) const {
    return _ghost_to_global[ghost_node];
  }

  [[nodiscard]] PEID ghost_owner(const NodeID ghost_node) const {
    return static_cast<PEID>(_ghost_owner[ghost_node]);
  }

private:
  NodeID _num_nodes;
  NodeID _num_ghost_nodes;

  bool _use_dense_global_to_ghost;
  growt::StaticGhostNodeMapping _sparse_global_to_ghost;

  RankCombinedBitVector<> _global_to_ghost_bitmap;
  CompactStaticArray<NodeID> _dense_global_to_ghost;

  CompactStaticArray<GlobalNodeID> _ghost_to_global;
  CompactStaticArray<UPEID> _ghost_owner;
};

class CompactGhostNodeMappingBuilder {
  SET_DEBUG(false);

  // @todo replace by growt hash table
  using GhostNodeMap = tbb::concurrent_hash_map<GlobalNodeID, NodeID>;

public:
  CompactGhostNodeMappingBuilder(
      const PEID rank, const StaticArray<GlobalNodeID> &node_distribution
  )
      : _num_nodes(static_cast<NodeID>(node_distribution[rank + 1] - node_distribution[rank])),
        _node_distribution(node_distribution.begin(), node_distribution.end()),
        _next_ghost_node(_num_nodes) {}

  NodeID new_ghost_node(const GlobalNodeID global_node) {
    GhostNodeMap::accessor entry;
    if (_global_to_ghost.insert(entry, global_node)) {
      const NodeID ghost_node = _next_ghost_node++;
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

  [[nodiscard]] CompactGhostNodeMapping finalize() {
    const NodeID num_ghost_nodes = _next_ghost_node - _num_nodes;
    const GlobalNodeID num_global_nodes = _node_distribution.back();
    const std::size_t num_processes = _node_distribution.size() - 1;

    RECORD("ghost_to_global")
    CompactStaticArray<GlobalNodeID> ghost_to_global(
        math::byte_width(num_global_nodes - 1), num_ghost_nodes
    );

    RECORD("ghost_owner")
    CompactStaticArray<UPEID> ghost_owner(math::byte_width(num_processes - 1), num_ghost_nodes);

    const auto foreach_global_to_ghost = [&](auto &&l) {
      for (const auto [global_node, local_node] : _global_to_ghost) {
        const NodeID local_ghost = local_node - _num_nodes;

        const auto owner_it =
            std::upper_bound(_node_distribution.begin() + 1, _node_distribution.end(), global_node);
        const auto owner =
            static_cast<PEID>(std::distance(_node_distribution.begin(), owner_it) - 1);

        l(global_node, local_node, local_ghost, owner);
      }
    };

    const std::size_t sparse_size =
        num_ghost_nodes * sizeof(growt::StaticGhostNodeMapping::atomic_slot_type);
    const std::size_t dense_size =
        num_global_nodes / 8 + num_ghost_nodes * math::byte_width(num_ghost_nodes - 1);

    if (sparse_size >= dense_size) {
      RankCombinedBitVector global_to_ghost_bitmap(_node_distribution.back());
      foreach_global_to_ghost([&](const GlobalNodeID global_node,
                                  const NodeID local_node,
                                  const NodeID local_ghost,
                                  const PEID owner) { global_to_ghost_bitmap.set(global_node); });
      global_to_ghost_bitmap.update();

      RECORD("dense_global_to_ghost")
      CompactStaticArray<NodeID> dense_global_to_ghost(
          math::byte_width(num_ghost_nodes - 1), num_ghost_nodes
      );
      foreach_global_to_ghost([&](const GlobalNodeID global_node,
                                  const NodeID local_node,
                                  const NodeID local_ghost,
                                  const PEID owner) {
        const std::size_t dense_index = global_to_ghost_bitmap.rank(global_node);
        dense_global_to_ghost.write(dense_index, local_ghost);

        ghost_to_global.write(local_ghost, global_node);
        ghost_owner.write(local_ghost, owner);
      });

      return CompactGhostNodeMapping(
          _num_nodes,
          num_ghost_nodes,
          std::move(global_to_ghost_bitmap),
          std::move(dense_global_to_ghost),
          std::move(ghost_to_global),
          std::move(ghost_owner)
      );
    } else {
      growt::StaticGhostNodeMapping global_to_ghost(num_ghost_nodes);
      foreach_global_to_ghost([&](const GlobalNodeID global_node,
                                  const NodeID local_node,
                                  const NodeID local_ghost,
                                  const PEID owner) {
        DBG << "Map global node " << global_node << " to local ghost node " << local_node;
        global_to_ghost.insert(global_node + 1, local_node);

        ghost_to_global.write(local_ghost, global_node);
        ghost_owner.write(local_ghost, owner);
      });

      return CompactGhostNodeMapping(
          _num_nodes,
          num_ghost_nodes,
          std::move(global_to_ghost),
          std::move(ghost_to_global),
          std::move(ghost_owner)
      );
    }
  }

private:
  NodeID _num_nodes;
  StaticArray<GlobalNodeID> _node_distribution;

  NodeID _next_ghost_node;
  GhostNodeMap _global_to_ghost;
};

namespace graph {
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

    RECORD("ghost_to_global") StaticArray<GlobalNodeID> ghost_to_global(ghost_n);
    RECORD("ghost_owner") StaticArray<PEID> ghost_owner(ghost_n);

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

    RECORD("global_to_ghost");
    RECORD_LOCAL_DATA_STRUCT(
        "growt::StaticGhostNodeMapping",
        global_to_ghost.capacity() * sizeof(growt::StaticGhostNodeMapping::atomic_slot_type)
    );

    return {
        .global_to_ghost = std::move(global_to_ghost),
        .ghost_to_global = std::move(ghost_to_global),
        .ghost_owner = std::move(ghost_owner),
    };
  }

private:
  StaticArray<GlobalNodeID> _node_distribution;
  NodeID _n;
  parallel::Atomic<NodeID> _next_ghost_node;
  GhostNodeMap _global_to_ghost;
};
} // namespace graph
} // namespace kaminpar::dist
