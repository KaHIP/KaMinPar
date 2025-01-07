/*******************************************************************************
 * Dynamic partition wrapper for a static graph.
 *
 * @file:   partitioned_graph.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <utility>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/graph_delegate.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

/*!
 * Extends a static graph with a dynamic graph partition.
 *
 * This class wraps a static graph and implements the same interface, delegating all calls to the
 * wrapped graph object. Additionally, it stores a mutable graph partition and implements a
 * thread-safe interface for accessing and changing the partition.
 *
 * If an object of this class is constructed without partition, all nodes are
 * marked unassigned, i.e., are placed in block `kInvalidBlockID`.
 */
template <typename GraphType> class GenericPartitionedGraph : public GraphDelegate<GraphType> {
public:
  using Graph = GraphType;

  using NodeID = typename Graph::NodeID;
  using NodeWeight = typename Graph::NodeWeight;
  using EdgeID = typename Graph::EdgeID;
  using EdgeWeight = typename Graph::EdgeWeight;
  using BlockID = ::kaminpar::shm::BlockID;
  using BlockWeight = ::kaminpar::shm::BlockWeight;

  // Tag for the sequential ctor.
  struct seq {};

  // Parallel ctor: use parallel loops to compute block weights.
  GenericPartitionedGraph(const Graph &graph, BlockID k, StaticArray<BlockID> partition = {})
      : GraphDelegate<Graph>(&graph),
        _k(k),
        _partition(std::move(partition)),
        _block_weights(k) {
    KASSERT(_partition.empty() || _partition.size() >= graph.n());

    if (graph.n() > 0 && _partition.empty()) {
      _partition.resize(graph.n(), kInvalidBlockID);
    }

    init_block_weights_par();
  }

  // Sequential ctor: use sequential loops to compute block weights.
  GenericPartitionedGraph(
      seq,
      const Graph &graph,
      BlockID k,
      StaticArray<BlockID> partition = {},
      StaticArray<BlockWeight> block_weights = {}
  )
      : GraphDelegate<Graph>(&graph),
        _k(k),
        _partition(std::move(partition)),
        _block_weights(std::move(block_weights)) {
    KASSERT(_partition.empty() || _partition.size() >= graph.n());
    KASSERT(_block_weights.empty() || _block_weights.size() >= k);

    if (graph.n() > 0 && _partition.empty()) {
      _partition.resize(graph.n(), kInvalidBlockID, static_array::seq);
    }

    if (k > 0 && _block_weights.empty()) {
      _block_weights.resize(k, static_array::seq);
    }

    init_block_weights_seq();
  }

  // Parallel + sequential ctor: take all data as input, do not initialize anything ourself.
  GenericPartitionedGraph(
      const Graph &graph,
      BlockID k,
      StaticArray<BlockID> partition,
      StaticArray<BlockWeight> block_weights
  )
      : GraphDelegate<Graph>(&graph),
        _k(k),
        _partition(std::move(partition)),
        _block_weights(std::move(block_weights)) {
    KASSERT(_partition.size() >= graph.n());
  }

  // Dummy ctor to make the class default-constructible for convenience.
  // @todo Should we get rid of this ctor?
  GenericPartitionedGraph() : GraphDelegate<Graph>(nullptr) {}

  GenericPartitionedGraph(const GenericPartitionedGraph &) = delete;
  GenericPartitionedGraph &operator=(const GenericPartitionedGraph &) = delete;

  GenericPartitionedGraph(GenericPartitionedGraph &&) noexcept = default;
  GenericPartitionedGraph &operator=(GenericPartitionedGraph &&other) noexcept = default;

  /**
   * Attempts to move node `u` from block `from` to block `to` while preserving the balance
   * constraint, i.e., the move will fail if it would increase the weight of `to` beyond
   * `max_weight`.
   *
   * This operation is thread-safe.
   *
   * @param u Node to be moved.
   * @param from Block the node is moved from (must be the current block of `u`).
   * @param to Block the node is moved to.
   * @param max_weight Maximum weight limit for block `to`.
   * @return Whether the node could be moved; if `false`, no change occurred.
   */
  [[nodiscard]] bool
  move(const NodeID u, const BlockID from, const BlockID to, const BlockWeight max_to_weight) {
    KASSERT(u < this->n());
    KASSERT(from < k());
    KASSERT(to < k());
    KASSERT(block(u) == from);
    KASSERT(from != to);

    if (move_block_weight(from, to, this->node_weight(u), max_to_weight)) {
      set_block<false>(u, to);
      return true;
    }

    return false;
  }

  /**
   * Move node `u` to block `to` (or assign it to the block if it does not
   * already belong to a different block) and optionally update block weights to reflect the move.
   *
   * In contrast to move(), this operation does not check whether the node move would lead to
   * violations of the balance constraint and might also be called for unassigned nodes.
   *
   * This operation is thread-safe.
   *
   * @tparam update_block_weight If set, atomically update the block weights.
   * @param u Node to be moved.
   * @param new_b Block the node is moved to.
   */
  template <bool update_block_weights = true> void set_block(const NodeID u, const BlockID to) {
    KASSERT(u < this->n(), "invalid node id " << u);
    KASSERT(to < k(), "invalid block id " << to << " for node " << u);

    if constexpr (update_block_weights) {
      const NodeWeight weight = this->node_weight(u);
      if (const BlockID from = block(u); from != kInvalidBlockID) {
        decrease_block_weight(from, weight);
      }
      increase_block_weight(to, weight);
    }

    __atomic_store_n(&_partition[u], to, __ATOMIC_RELAXED);
  }

  /**
   * Shift weight from block `from` to block `to`. This operation will fail if the weight shift
   * would overload `to`.
   *
   * This operation is thread-safe.
   *
   * @param from Block to move weight from.
   * @param to Block to move weight to.
   * @param delta Amount of weight to be moved.
   * @param max_weight Maximum weight limit for block `to`.
   * @return Whether the weight could be moved; if `false`, no change occurred.
   */
  [[nodiscard]] bool move_block_weight(
      const BlockID from, const BlockID to, const BlockWeight delta, const BlockWeight max_to_weight
  ) {
    for (BlockWeight new_weight = block_weight(to); new_weight + delta <= max_to_weight;) {
      if (__atomic_compare_exchange_n(
              &_block_weights[to],
              &new_weight,
              new_weight + delta,
              false,
              __ATOMIC_RELAXED,
              __ATOMIC_RELAXED
          )) {
        decrease_block_weight(from, delta);
        return true;
      }
    }

    return false;
  }

  //
  // Raw partition access
  //

  [[nodiscard]] inline const StaticArray<BlockID> &raw_partition() const {
    return _partition;
  }

  [[nodiscard]] inline StaticArray<BlockID> &&take_raw_partition() {
    return std::move(_partition);
  }

  [[nodiscard]] inline const StaticArray<BlockWeight> &raw_block_weights() const {
    return _block_weights;
  }

  [[nodiscard]] inline StaticArray<BlockWeight> &&take_raw_block_weights() {
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

  void increase_block_weight(const BlockID b, const BlockWeight by) {
    KASSERT(b < k());
    __atomic_fetch_add(&_block_weights[b], by, __ATOMIC_RELAXED);
  }

  void decrease_block_weight(const BlockID b, const BlockWeight by) {
    KASSERT(b < k());
    __atomic_fetch_sub(&_block_weights[b], by, __ATOMIC_RELAXED);
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
    KASSERT(u < this->n());
    return __atomic_load_n(&_partition[u], __ATOMIC_RELAXED);
  }

private:
  void init_block_weights_par() {
    if (k() >= 65536) {
      this->pfor_nodes([&](const NodeID u) {
        __atomic_fetch_add(&_block_weights[block(u)], this->node_weight(u), __ATOMIC_RELAXED);
      });
      return;
    }

    tbb::enumerable_thread_specific<StaticArray<BlockWeight>> block_weights_ets([&] {
      return StaticArray<BlockWeight>(k());
    });

    tbb::parallel_for(
        tbb::blocked_range<NodeID>(0, this->n()),
        [&](const tbb::blocked_range<NodeID> &r) {
          auto &block_weights = block_weights_ets.local();
          for (NodeID u = r.begin(); u != r.end(); ++u) {
            if (const BlockID b = block(u); b != kInvalidBlockID) {
              block_weights[b] += this->node_weight(u);
            }
          }
        }
    );

    tbb::parallel_for<BlockID>(0, k(), [&](const BlockID b) {
      BlockWeight sum = 0;
      for (const StaticArray<BlockWeight> &block_weights : block_weights_ets) {
        sum += block_weights[b];
      }
      _block_weights[b] = sum;
    });
  }

  void init_block_weights_seq() {
    for (const NodeID u : this->nodes()) {
      if (const BlockID b = block(u); b != kInvalidBlockID) {
        _block_weights[b] += this->node_weight(u);
      }
    }
  }

  BlockID _k = 0;
  StaticArray<BlockID> _partition;
  StaticArray<BlockWeight> _block_weights;
};

using PartitionedGraph = GenericPartitionedGraph<Graph>;
using PartitionedCSRGraph = GenericPartitionedGraph<CSRGraph>;

} // namespace kaminpar::shm
