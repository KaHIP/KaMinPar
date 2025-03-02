/*******************************************************************************
 * Stores changes to a static graph partition.
 *
 * @file:   delta_partitioned_graph.h
 * @author: Daniel Seemaier
 * @date:   15.03.2023
 ******************************************************************************/
#pragma once

#include <span>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/ranges.h"

namespace kaminpar::shm {

class DeltaPartitionedGraph {
public:
  DeltaPartitionedGraph(const PartitionedGraph *p_graph) : _p_graph(p_graph) {
    _node_weights = reified(*_p_graph, [&](const auto &g) { return g.raw_node_weights().view(); });
    _block_weights_delta.resize(_p_graph->k());
  }

  [[nodiscard]] const PartitionedGraph &p_graph() const {
    return *_p_graph;
  }

  [[nodiscard]] inline const Graph &graph() const {
    return _p_graph->graph();
  }

  [[nodiscard]] inline NodeWeight node_weight(const NodeID u) const {
    return _node_weights.empty() ? 1 : _node_weights[u];
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
    const auto *it = _partition_delta.get_if_contained(node);
    return (it != _partition_delta.end()) ? *it : _p_graph->block(node);
  }

  template <bool update_block_weight = true>
  void set_block(const NodeID node, const BlockID new_block) {
    KASSERT(node < _p_graph->graph().n(), "invalid node id " << node);
    KASSERT(new_block < k(), "invalid block id " << new_block << " for node " << node);

    if constexpr (update_block_weight) {
      const BlockID old_block = block(node);
      KASSERT(old_block < k());

      const NodeWeight w = node_weight(node);
      _block_weights_delta[old_block] -= w;
      _block_weights_delta[new_block] += w;
    }

    _partition_delta[node] = new_block;
  }

  [[nodiscard]] inline BlockWeight block_weight(const BlockID block) const {
    return _p_graph->block_weight(block) + _block_weights_delta[block];
  }

  template <typename Lambda> void for_each(Lambda &&lambda) {
    _partition_delta.for_each(std::forward<Lambda>(lambda));
  }

  [[nodiscard]] std::size_t size() const {
    return _partition_delta.size();
  }

  void clear() {
    std::fill(_block_weights_delta.begin(), _block_weights_delta.end(), 0);
    _partition_delta.clear();
  }

private:
  const PartitionedGraph *_p_graph;
  std::span<const NodeWeight> _node_weights;

  ScalableVector<BlockWeight> _block_weights_delta;
  DynamicFlatMap<NodeID, BlockID> _partition_delta;
};

template <typename Lambda> decltype(auto) reified(DeltaPartitionedGraph &p_graph, Lambda &&l) {
  return reified(p_graph.graph(), std::forward<Lambda>(l));
}

template <typename Lambda>
decltype(auto) reified(const DeltaPartitionedGraph &p_graph, Lambda &&l) {
  return reified(p_graph.graph(), std::forward<Lambda>(l));
}

} // namespace kaminpar::shm
