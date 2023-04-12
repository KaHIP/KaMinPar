/*******************************************************************************
 * @file:   delta_partitioned_graph.h
 * @author: Daniel Seemaier
 * @date:   15.03.2023
 * @brief:  Stores changes to a static partitioned graph.
 ******************************************************************************/
#pragma once

#include <google/dense_hash_map>

#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"
#include "kaminpar/definitions.h"

#include "common/ranges.h"

namespace kaminpar::shm {
class DeltaPartitionedGraph : public GraphDelegate {
public:
  DeltaPartitionedGraph(const PartitionedGraph *p_graph)
      : GraphDelegate(&p_graph->graph()),
        _p_graph(p_graph) {
    _block_weights_delta.set_deleted_key(kInvalidBlockID);
    _block_weights_delta.set_empty_key(kInvalidBlockID - 1);
    _partition_delta.set_deleted_key(kInvalidNodeID);
    _partition_delta.set_empty_key(kInvalidNodeID - 1);
  }

  [[nodiscard]] const PartitionedGraph &p_graph() const {
    return *_p_graph;
  }

  [[nodiscard]] inline NodeID n() const {
    return _p_graph->n();
  }
  [[nodiscard]] inline BlockID k() const {
    return _p_graph->k();
  }

  [[nodiscard]] inline IotaRange<BlockID> blocks() const {
    return _p_graph->blocks();
  }

  template <typename Lambda> inline void pfor_blocks(Lambda &&l) const {
    tbb::parallel_for(static_cast<BlockID>(0), k(), std::forward<Lambda>(l));
  }

  [[nodiscard]] inline BlockID block(const NodeID u) const {
    const auto it = _partition_delta.find(u);
    if (it != _partition_delta.end()) {
      return it->second;
    }

    return _p_graph->block(u);
  }

  template <bool update_block_weight = true>
  void set_block(const NodeID u, const BlockID new_b) {
    KASSERT(u < n(), "invalid node id " << u);
    KASSERT(new_b < k(), "invalid block id " << new_b << " for node " << u);

    if constexpr (update_block_weight) {
      if (block(u) != kInvalidBlockID) {
        _block_weights_delta[block(u)] -= node_weight(u);
      }
      _block_weights_delta[new_b] += node_weight(u);
    }

    _partition_delta[u] = new_b;
  }

  [[nodiscard]] inline NodeWeight block_weight(const BlockID b) const {
    NodeWeight delta = 0;
    const auto it = _block_weights_delta.find(b);
    if (it != _block_weights_delta.end()) {
      delta = it->second;
    }

    return _p_graph->block_weight(b) + delta;
  }

  const auto &delta() const {
    return _partition_delta;
  }

  void clear() {
    _block_weights_delta.clear();
    _partition_delta.clear();
  }

private:
  const PartitionedGraph *_p_graph;

  google::dense_hash_map<BlockID, NodeWeight> _block_weights_delta;
  google::dense_hash_map<NodeID, BlockID> _partition_delta;
};
} // namespace kaminpar::shm

