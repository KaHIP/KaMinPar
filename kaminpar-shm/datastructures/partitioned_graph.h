/*******************************************************************************
 * Dynamic partition wrapper for a static graph.
 *
 * @file:   partitioned_graph.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <utility>
#include <vector>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/graph_delegate.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/parallel/atomic.h"

namespace kaminpar::shm {
/*!
 * Extends a kaminpar::Graph with a graph partition.
 *
 * This class implements the same member functions as kaminpar::Graph plus some
 * more that only concern the graph partition. Functions that are also
 * implemented in kaminpar::Graph are delegated to the wrapped object.
 *
 * If an object of this class is constructed without partition, all nodes are
 * marked unassigned, i.e., are placed in block `kInvalidBlockID`.
 */
class PartitionedGraph : public GraphDelegate {
public:
  using NodeID = Graph::NodeID;
  using NodeWeight = Graph::NodeWeight;
  using EdgeID = Graph::EdgeID;
  using EdgeWeight = Graph::EdgeWeight;
  using BlockID = ::kaminpar::shm::BlockID;
  using BlockWeight = ::kaminpar::shm::BlockWeight;

  PartitionedGraph(const Graph &graph, BlockID k, StaticArray<BlockID> partition = {});

  PartitionedGraph(
      tag::Sequential, const Graph &graph, BlockID k, StaticArray<BlockID> partition = {}
  );

  PartitionedGraph() : GraphDelegate(nullptr) {}

  PartitionedGraph(const PartitionedGraph &) = delete;
  PartitionedGraph &operator=(const PartitionedGraph &) = delete;

  PartitionedGraph(PartitionedGraph &&) noexcept = default;
  PartitionedGraph &operator=(PartitionedGraph &&other) noexcept = default;

  ~PartitionedGraph() = default;

  bool try_balanced_move(
      const NodeID u, const BlockID from, const BlockID to, const BlockWeight max_weight
  ) {
    KASSERT(block(u) == from);
    KASSERT(from != to);

    if (try_move_block_weight(from, to, node_weight(u), max_weight)) {
      set_block<false>(u, to);
      return true;
    }

    return false;
  }

  bool try_balanced_move(const NodeID u, const BlockID to, const BlockWeight max_weight) {
    return try_balanced_move(u, block(u), to, max_weight);
  }

  /**
   * Move node `u` to block `new_b` (or assign it to the block if it does not
   * already belong to a different block) and update block weights accordingly
   * (optional).
   *
   * @tparam update_block_weight If set, atomically update the block weights.
   * @param u Node to be moved.
   * @param new_b Block the node is moved to.
   */
  template <bool update_block_weights = true> void set_block(const NodeID u, const BlockID to) {
    KASSERT(u < n(), "invalid node id " << u);
    KASSERT(to < k(), "invalid block id " << to << " for node " << u);

    if constexpr (update_block_weights) {
      const NodeWeight weight = node_weight(u);
      if (const BlockID from = block(u); from != kInvalidBlockID) {
        __atomic_fetch_sub(&_block_weights[from], weight, __ATOMIC_RELAXED);
      }
      __atomic_fetch_add(&_block_weights[to], weight, __ATOMIC_RELAXED);
    }

    __atomic_store_n(&_partition[u], to, __ATOMIC_RELAXED);
  }

  /**
   * Attempt to move weight from block `from` to block `to` subject to the
   * weight constraint `max_weight` using CAS operations.
   *
   * @param from Block to move weight from.
   * @param to Block to move weight to.
   * @param delta Amount of weight to be moved, i.e., subtracted from `from` and
   * added to `to`.
   * @param max_weight Weight constraint for block `to`.
   * @return Whether the weight could be moved; if `false`, no change occurred.
   */
  bool try_move_block_weight(
      const BlockID from, const BlockID to, const BlockWeight delta, const BlockWeight max_weight
  ) {
    BlockWeight new_weight = block_weight(to);
    bool success = false;

    while (new_weight + delta <= max_weight) {
      if (__atomic_compare_exchange_n(
              &_block_weights[to],
              &new_weight,
              new_weight + delta,
              false,
              __ATOMIC_RELAXED,
              __ATOMIC_RELAXED
          )) {
        success = true;
        break;
      }
    }

    if (success) {
      __atomic_fetch_sub(&_block_weights[from], delta, __ATOMIC_RELAXED);
    }
    return success;
  }

  //
  // Raw partition access
  //

  [[nodiscard]] inline const StaticArray<BlockID> &raw_partition() const {
    return _partition;
  }

  [[nodiscard]] inline StaticArray<BlockID> &&take_partition() {
    return std::move(_partition);
  }

  [[nodiscard]] inline const StaticArray<BlockWeight> &raw_block_weights() const {
    return _block_weights;
  }

  [[nodiscard]] inline StaticArray<BlockWeight> &&take_block_weights() {
    return std::move(_block_weights);
  }

  //
  // Block weights
  //

  [[nodiscard]] inline BlockWeight block_weight(const BlockID b) const {
    KASSERT(b < k());
    return __atomic_load_n(&_block_weights[b], __ATOMIC_RELAXED);
  }

  void set_block_weight(const BlockID b, const BlockWeight weight) {
    KASSERT(b < k());
    __atomic_store_n(&_block_weights[b], weight, __ATOMIC_RELAXED);
  }

  //
  // Parallel iteration
  //

  template <typename Lambda> inline void pfor_blocks(Lambda &&l) const {
    tbb::parallel_for(static_cast<BlockID>(0), k(), std::forward<Lambda>(l));
  }

  //
  // Sequential iteration
  //

  [[nodiscard]] inline IotaRange<BlockID> blocks() const {
    return {static_cast<BlockID>(0), k()};
  }

  //
  // Partition access
  //

  [[nodiscard]] inline BlockID k() const {
    return _k;
  }

  [[nodiscard]] inline BlockID block(const NodeID u) const {
    return __atomic_load_n(&_partition[u], __ATOMIC_RELAXED);
  }

  //
  // Final k's
  //

private:
  void init_block_weights_par();
  void init_block_weights_seq();

  BlockID _k = 0;
  StaticArray<BlockID> _partition;
  StaticArray<BlockWeight> _block_weights;
};
} // namespace kaminpar::shm
