/*******************************************************************************
 * @file:   partitioned_graph.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Static graph with a dynamic partition.
 ******************************************************************************/
#pragma once

#include "kaminpar/datastructures/graph.h"

#include "common/parallel/atomic.h"

namespace kaminpar::shm {
using BlockArray = StaticArray<BlockID>;
using BlockWeightArray = StaticArray<parallel::Atomic<BlockWeight>>;

class GreedyBalancer;

struct NoBlockWeights {};
constexpr NoBlockWeights no_block_weights;

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
  friend GreedyBalancer;

public:
  using NodeID = Graph::NodeID;
  using NodeWeight = Graph::NodeWeight;
  using EdgeID = Graph::EdgeID;
  using EdgeWeight = Graph::EdgeWeight;
  using BlockID = ::kaminpar::shm::BlockID;
  using BlockWeight = ::kaminpar::shm::BlockWeight;

  PartitionedGraph(
      const Graph &graph,
      BlockID k,
      StaticArray<BlockID> partition = {},
      std::vector<BlockID> final_k = {}
  );

  PartitionedGraph(
      tag::Sequential,
      const Graph &graph,
      BlockID k,
      StaticArray<BlockID> partition = {},
      std::vector<BlockID> final_k = {}
  );

  PartitionedGraph(NoBlockWeights, const Graph &graph, BlockID k, StaticArray<BlockID> partition);

  PartitionedGraph() : GraphDelegate(nullptr) {}

  PartitionedGraph(const PartitionedGraph &) = delete;
  PartitionedGraph &operator=(const PartitionedGraph &) = delete;

  PartitionedGraph(PartitionedGraph &&) noexcept = default;
  PartitionedGraph &operator=(PartitionedGraph &&other) noexcept = default;

  /**
   * Move node `u` to block `new_b` (or assign it to the block if it does not
   * already belong to a different block) and update block weights accordingly
   * (optional).
   *
   * @tparam update_block_weight If set, atomically update the block weights.
   * @param u Node to be moved.
   * @param new_b Block the node is moved to.
   */
  template <bool update_block_weight = true> void set_block(const NodeID u, const BlockID new_b) {
    KASSERT(u < n(), "invalid node id " << u);
    KASSERT(new_b < k(), "invalid block id " << new_b << " for node " << u);

    if constexpr (update_block_weight) {
      if (block(u) != kInvalidBlockID) {
        _block_weights[block(u)] -= node_weight(u);
      }
      _block_weights[new_b] += node_weight(u);
    }

    __atomic_store_n(&_partition[u], new_b, __ATOMIC_RELAXED);
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
      if (_block_weights[to].compare_exchange_weak(
              new_weight, new_weight + delta, std::memory_order_relaxed
          )) {
        success = true;
        break;
      }
    }

    if (success) {
      _block_weights[from].fetch_sub(delta, std::memory_order_relaxed);
    }
    return success;
  }

  void change_k(BlockID new_k);
  void reinit_block_weights();

  // clang-format off
  [[nodiscard]] inline IotaRange<BlockID> blocks() const { return IotaRange(static_cast<BlockID>(0), k()); }
  [[nodiscard]] inline BlockID block(const NodeID u) const { return __atomic_load_n(&_partition[u], __ATOMIC_RELAXED); }
  template <typename Lambda> inline void pfor_blocks(Lambda &&l) const { tbb::parallel_for(static_cast<BlockID>(0), k(), std::forward<Lambda>(l)); }
  [[nodiscard]] inline NodeWeight block_weight(const BlockID b) const { KASSERT(b < k()); return _block_weights[b]; }
  void set_block_weight(const BlockID b, const BlockWeight weight) { KASSERT(b < k()); _block_weights[b] = weight; }
  [[nodiscard]] inline const auto &block_weights() const { return _block_weights; }
  [[nodiscard]] inline auto &&take_block_weights() { return std::move(_block_weights); }
  [[nodiscard]] inline BlockID heaviest_block() const { return std::max_element(_block_weights.begin(), _block_weights.end()) - _block_weights.begin(); }
  [[nodiscard]] inline BlockID lightest_block() const { return std::min_element(_block_weights.begin(), _block_weights.end()) - _block_weights.begin(); }
  [[nodiscard]] inline BlockID k() const { return _k; }
  [[nodiscard]] inline const auto &partition() const { return _partition; }
  [[nodiscard]] inline auto &&take_partition() { return std::move(_partition); }
  [[nodiscard]] inline BlockID final_k(const BlockID b) const { return _final_k[b]; }
  [[nodiscard]] inline const std::vector<BlockID> &final_ks() const { return _final_k; }
  [[nodiscard]] inline std::vector<BlockID> &&take_final_k() { return std::move(_final_k); }
  inline void set_final_k(const BlockID b, const BlockID final_k) { _final_k[b] = final_k; }
  inline void set_final_ks(std::vector<BlockID> final_ks) { _final_k = std::move(final_ks); }
  // clang-format on

private:
  void init_block_weights_par();
  void init_block_weights_seq();

  BlockID _k;
  StaticArray<BlockID> _partition;
  StaticArray<parallel::Atomic<NodeWeight>> _block_weights;

  //! For each block in the current partition, this is the number of blocks that
  //! we want to split the block in the final partition. For instance, after the
  //! first bisection, this might be {_k / 2, _k / 2}, although other values are
  //! possible if _k is not a power of 2.
  std::vector<BlockID> _final_k;
};
} // namespace kaminpar::shm
