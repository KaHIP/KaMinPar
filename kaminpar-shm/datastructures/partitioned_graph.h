/**
 * @file partitioned_graph.cc
 * @brief A dynamic graph partition on top of a static graph.
 */
#pragma once

#include <span>
#include <utility>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/ranges.h"

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
template <typename GraphType> class GenericPartitionedGraph {
public:
  using Graph = GraphType;

  using NodeID = typename Graph::NodeID;
  using NodeWeight = typename Graph::NodeWeight;
  using EdgeID = typename Graph::EdgeID;
  using EdgeWeight = typename Graph::EdgeWeight;
  using BlockID = ::kaminpar::shm::BlockID;
  using BlockWeight = ::kaminpar::shm::BlockWeight;

  // Tags for sequential vs parallel initialization
  struct seq {};
  struct par {};

  // Maximum k for which block weights are spreaded to individual cache lines
  constexpr static BlockID kDenseBlockWeightsThreshold = 256;

  // Parallel ctor: use parallel loops to compute block weights.
  GenericPartitionedGraph(
      const Graph &graph,
      const BlockID k,
      StaticArray<BlockID> partition,
      StaticArray<BlockWeight> block_weights = {}
  );

  // Sequential ctor: use sequential loops to compute block weights.
  GenericPartitionedGraph(
      seq,
      const Graph &graph,
      const BlockID k,
      StaticArray<BlockID> partition,
      StaticArray<BlockWeight> block_weights = {}
  );

  // Dummy ctor to make the class default-constructible for convenience.
  GenericPartitionedGraph() {}

  GenericPartitionedGraph(const GenericPartitionedGraph &) = delete;
  GenericPartitionedGraph &operator=(const GenericPartitionedGraph &) = delete;

  GenericPartitionedGraph(GenericPartitionedGraph &&) noexcept = default;
  GenericPartitionedGraph &operator=(GenericPartitionedGraph &&other) noexcept = default;

  /**
   * Attempt to move node `u` from block `from` to block `to` while preserving the balance
   * constraint, i.e., the move will fail if it would increase the weight of `to` beyond
   * `max_to_weight`, or the weight of `from` below `min_from_weight`.
   *
   * This operation is thread-safe and guarantees that the `to` block will not be overloaded.
   * The `to` block might become underloaded due to race conditions.
   *
   * @param u Node to be moved.
   * @param from Block the node is moved from (must be the current block of `u`).
   * @param to Block the node is moved to.
   * @param max_to_weight Maximum weight for block `to`.
   * @param min_from_weight Minimum weight for block `from`.
   *
   * @return Whether the node could be moved; if `false`, no change occurred.
   */
  [[nodiscard]] inline bool move(
      const NodeID u,
      const BlockID from,
      const BlockID to,
      const BlockWeight max_to_weight,
      const BlockWeight min_from_weight = 0
  ) {
    KASSERT(u < _graph->n());
    KASSERT(from != to);
    KASSERT(from < k() && to < k());
    KASSERT(block(u) == from);

    if (move_block_weight(from, to, node_weight(u), max_to_weight, min_from_weight)) {
      set_block<false>(u, to);
      return true;
    }

    return false;
  }

  /**
   * Move node `u` to block `to` and update block weights to reflect the move (optionally). In
   * contrast to `move()`, this operation does not enforce balance constraints. Thus, it will always
   * succeed.
   *
   * This operation is thread-safe.
   *
   * @tparam update_block_weight If set, atomically update the block weights.
   * @param u Node to be moved.
   * @param to Block the node is moved to.
   */
  template <bool update_block_weights = true>
  inline void set_block(const NodeID u, const BlockID to) {
    KASSERT(u < _graph->n(), "invalid node id " << u);
    KASSERT(to < k(), "invalid block id " << to << " for node " << u);
    KASSERT(!update_block_weights || block(u) != kInvalidBlockID);

    if constexpr (update_block_weights) {
      const BlockID from = block(u);
      const NodeWeight weight = node_weight(u);

      if (use_dense_block_weights()) {
        __atomic_fetch_sub(&_dense_block_weights[from], weight, __ATOMIC_RELAXED);
        __atomic_fetch_add(&_dense_block_weights[to], weight, __ATOMIC_RELAXED);
      } else { // use_aligned_block_weights()
        _diverged_block_weights = true;
        __atomic_fetch_sub(&_aligned_block_weights[from].value, weight, __ATOMIC_RELAXED);
        __atomic_fetch_add(&_aligned_block_weights[to].value, weight, __ATOMIC_RELAXED);
      }
    }

    __atomic_store_n(&_partition[u], to, __ATOMIC_RELAXED);
  }

  /**
   * Shift the given weight delta from block `from` to block `to`. This operation will fail if the
   * weight shift would overload block `to` or underload block `from`.
   *
   * This operation is thread-safe. In case of failure, it is possible that block `to` becomes
   * underloaded due to race conditions.
   *
   * @param from Block to move weight from.
   * @param to Block to move weight to.
   * @param delta Amount of weight to be moved.
   * @param max_to_weight Maximum weight for block `to`.
   * @param min_from_weight Minimum weight for block `from`.
   *
   * @return Whether the weight could be moved; if `false`, no change occurred.
   */
  [[nodiscard]] inline bool move_block_weight(
      const BlockID from,
      const BlockID to,
      const BlockWeight delta,
      const BlockWeight max_to_weight,
      const BlockWeight min_from_weight = 0
  ) {
    if (use_dense_block_weights()) {
      return move_block_weight_impl(
          _dense_block_weights,
          [](auto &entry) { return &entry; },
          from,
          to,
          delta,
          max_to_weight,
          min_from_weight
      );
    } else { // use_aligned_block_weights()
      _diverged_block_weights = true;
      return move_block_weight_impl(
          _aligned_block_weights,
          [](auto &entry) { return &entry.value; },
          from,
          to,
          delta,
          max_to_weight,
          min_from_weight
      );
    }
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
    sync_dense_and_aligned_block_weights();
    return _dense_block_weights;
  }

  [[nodiscard]] inline StaticArray<BlockWeight> &&take_raw_block_weights() {
    sync_dense_and_aligned_block_weights();
    return std::move(_dense_block_weights);
  }

  //
  // Block weights
  //

  [[nodiscard]] inline BlockWeight block_weight(const BlockID b) const {
    KASSERT(b < k());

    if (use_dense_block_weights()) {
      return __atomic_load_n(&_dense_block_weights[b], __ATOMIC_RELAXED);
    } else { // use_aligned_block_weights()
      return __atomic_load_n(&_aligned_block_weights[b].value, __ATOMIC_RELAXED);
    }
  }

  //
  // Parallel iteration
  //

  template <typename Lambda> inline void pfor_blocks(Lambda &&l) const {
    tbb::parallel_for<BlockID>(0, k(), std::forward<Lambda>(l));
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
    KASSERT(u < _graph->n());

    return __atomic_load_n(&_partition[u], __ATOMIC_RELAXED);
  }

  [[nodiscard]] inline NodeID n() const {
    return _graph->n();
  }

  [[nodiscard]] inline EdgeID m() const {
    return _graph->m();
  }

  [[nodiscard]] NodeWeight node_weight(const NodeID u) const {
    return _node_weights.empty() ? 1 : _node_weights[u];
  }

  [[nodiscard]] inline const Graph &graph() const {
    return *_graph;
  }

  void reinit_block_weights(bool sequentially = false);

private:
  template <typename ValuePtrGetter>
  [[nodiscard]] bool move_block_weight_impl(
      auto &block_weights_vec,
      ValuePtrGetter &&ptr,
      const BlockID from,
      const BlockID to,
      const BlockWeight delta,
      const BlockWeight max_to_weight,
      const BlockWeight min_from_weight
  ) {
    for (BlockWeight new_weight = block_weight(to); new_weight + delta <= max_to_weight;) {
      if (__atomic_compare_exchange_n(
              ptr(block_weights_vec[to]),
              &new_weight,
              new_weight + delta,
              false,
              __ATOMIC_RELAXED,
              __ATOMIC_RELAXED
          )) {
        if (__atomic_sub_fetch(ptr(block_weights_vec[from]), delta, __ATOMIC_RELAXED) >=
            min_from_weight) {
          return true;
        } else {
          __atomic_fetch_add(ptr(block_weights_vec[from]), delta, __ATOMIC_RELAXED);
          __atomic_fetch_sub(ptr(block_weights_vec[to]), delta, __ATOMIC_RELAXED);
          return false;
        }
      }
    }

    return false;
  }

  [[nodiscard]] inline bool use_aligned_block_weights() const {
    return _k <= kDenseBlockWeightsThreshold;
  }

  [[nodiscard]] inline bool use_dense_block_weights() const {
    return !use_aligned_block_weights();
  }

  void sync_dense_and_aligned_block_weights() const;

  void init_block_weights(bool sequentially);

  void init_node_weights();

  const Graph *_graph = nullptr;
  std::span<const NodeWeight> _node_weights = {};

  BlockID _k = 0;
  StaticArray<BlockID> _partition = {};

  struct alignas(64) AlignedBlockWeight {
    BlockWeight value;
  };

  mutable bool _diverged_block_weights = false;
  mutable StaticArray<BlockWeight> _dense_block_weights = {};
  StaticArray<AlignedBlockWeight> _aligned_block_weights = {};
};

using PartitionedGraph = GenericPartitionedGraph<Graph>;
using PartitionedCSRGraph = GenericPartitionedGraph<CSRGraph>;

template <typename Lambda> decltype(auto) reified(PartitionedGraph &p_graph, Lambda &&l) {
  return reified(p_graph.graph(), std::forward<Lambda>(l));
}

template <typename Lambda> decltype(auto) reified(const PartitionedGraph &p_graph, Lambda &&l) {
  return reified(p_graph.graph(), std::forward<Lambda>(l));
}

template <typename Lambda1, typename Lambda2>
decltype(auto) reified(PartitionedGraph &p_graph, Lambda1 &&l1, Lambda2 &&l2) {
  return reified(p_graph.graph(), std::forward<Lambda1>(l1), std::forward<Lambda2>(l2));
}

template <typename Lambda1, typename Lambda2>
decltype(auto) reified(const PartitionedGraph &p_graph, Lambda1 &&l1, Lambda2 &&l2) {
  return reified(p_graph.graph(), std::forward<Lambda1>(l1), std::forward<Lambda2>(l2));
}

} // namespace kaminpar::shm
