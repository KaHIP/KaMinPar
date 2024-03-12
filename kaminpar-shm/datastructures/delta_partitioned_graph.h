/*******************************************************************************
 * Stores changes to a static partitioned graph.
 *
 * @file:   delta_partitioned_graph.h
 * @author: Daniel Seemaier
 * @date:   15.03.2023
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/datastructures/graph_delegate.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/ranges.h"

namespace kaminpar::shm {
template <
    // If false, block(NodeID) may only be called on nodes that were not moved.
    bool allow_read_after_move = true,
    // If false, store the block weight changes in a vector of size k, otherwise
    // use a hash map.
    bool compact_block_weight_delta = false>
class GenericDeltaPartitionedGraph : public GraphDelegate {
  struct DeltaEntry {
    NodeID node;
    BlockID block;
  };

public:
  constexpr static bool kAllowsReadAfterMove = allow_read_after_move;

  GenericDeltaPartitionedGraph(const PartitionedGraph *p_graph)
      : GraphDelegate(&p_graph->graph()),
        _p_graph(p_graph) {
    if constexpr (!compact_block_weight_delta) {
      _block_weights_delta.resize(_p_graph->k());
    }
  }

  [[nodiscard]] const PartitionedGraph &p_graph() const {
    return *_p_graph;
  }

  [[nodiscard]] inline BlockID k() const {
    return _p_graph->k();
  }

  [[nodiscard]] inline IotaRange<BlockID> blocks() const {
    return _p_graph->blocks();
  }

  template <typename Lambda> inline void pfor_blocks(Lambda &&lambda) const {
    tbb::parallel_for(static_cast<BlockID>(0), k(), std::forward<Lambda>(lambda));
  }

  [[nodiscard]] inline BlockID block(const NodeID node) const {
    if constexpr (allow_read_after_move) {
      const auto *it = _partition_delta.get_if_contained(node);
      return (it != _partition_delta.end()) ? *it : _p_graph->block(node);
    } else {
      KASSERT(
          std::find_if(
              _partition_delta.begin(),
              _partition_delta.end(),
              [&](const DeltaEntry &entry) { return entry.node == node; }
          ) == _partition_delta.end(),
          "node " << node << " was moved, access illegal",
          assert::heavy
      );

      return _p_graph->block(node);
    }
  }

  template <bool update_block_weight = true>
  void set_block(const NodeID node, const BlockID new_block) {
    KASSERT(node < n(), "invalid node id " << node);
    KASSERT(new_block < k(), "invalid block id " << new_block << " for node " << node);

    if constexpr (update_block_weight) {
      const BlockID old_block = block(node);
      KASSERT(old_block < k());

      _block_weights_delta[old_block] -= node_weight(node);
      _block_weights_delta[new_block] += node_weight(node);
    }

    if constexpr (allow_read_after_move) {
      _partition_delta[node] = new_block;
    } else {
      KASSERT(
          std::find_if(
              _partition_delta.begin(),
              _partition_delta.end(),
              [&](const DeltaEntry &entry) { return entry.node == node; }
          ) == _partition_delta.end(),
          "node " << node << " already in delta",
          assert::heavy
      );

      _partition_delta.push_back({node, new_block});
    }
  }

  [[nodiscard]] inline BlockWeight block_weight(const BlockID block) const {
    BlockWeight delta = 0;

    if constexpr (compact_block_weight_delta) {
      const auto *it = _block_weights_delta.get_if_contained(block);
      if (it != _block_weights_delta.end()) {
        delta = *it;
      }
    } else {
      delta = _block_weights_delta[block];
    }

    return _p_graph->block_weight(block) + delta;
  }

  template <typename Lambda> void for_each(Lambda &&lambda) {
    if constexpr (allow_read_after_move) {
      _partition_delta.for_each(std::forward<Lambda>(lambda));
    } else {
      for (const auto &[moved_node, moved_to] : _partition_delta) {
        lambda(moved_node, moved_to);
      }
    }
  }

  std::size_t size() const {
    return _partition_delta.size();
  }

  void clear() {
    if constexpr (compact_block_weight_delta) {
      _block_weights_delta.clear();
    } else {
      std::fill(_block_weights_delta.begin(), _block_weights_delta.end(), 0);
    }

    _partition_delta.clear();
  }

private:
  const PartitionedGraph *_p_graph;

  // Depending on the configuration, use a hash map to be memory efficient,
  // otherwise store the block weight deltas in vector (i.e., O(P * k) memory).
  std::conditional_t<
      compact_block_weight_delta,
      DynamicFlatMap<BlockID, BlockWeight>,
      scalable_vector<BlockWeight>>
      _block_weights_delta;

  // If we need random access to the partition delta, use a hash map. Otherwise,
  // we can just store the moves in a vector.
  std::conditional_t<
      allow_read_after_move,
      DynamicFlatMap<NodeID, BlockID>,
      scalable_vector<DeltaEntry>>
      _partition_delta;
};
} // namespace kaminpar::shm
