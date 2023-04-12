/*******************************************************************************
 * @file:   fm_refiner.h
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 * @brief:  Parallel k-way FM refinement algorithm.
 ******************************************************************************/
#pragma once

#include <cmath>

#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>

#include "kaminpar/context.h"
#include "kaminpar/datastructures/delta_partitioned_graph.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"
#include "kaminpar/refinement/gain_cache.h"
#include "kaminpar/refinement/refiner.h"

#include "common/datastructures/binary_heap.h"
#include "common/datastructures/marker.h"
#include "common/noinit_vector.h"
#include "common/parallel/atomic.h"

namespace kaminpar::shm {
class FMRefiner : public Refiner {
public:
  FMRefiner(const Context &ctx);

  FMRefiner(const FMRefiner &) = delete;
  FMRefiner &operator=(const FMRefiner &) = delete;

  FMRefiner(FMRefiner &&) noexcept = default;
  FMRefiner &operator=(FMRefiner &&) noexcept = default;

  void initialize(const Graph &graph) final;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) final;

  [[nodiscard]] EdgeWeight expected_total_gain() const final {
    return 0;
  }

private:
  bool run_localized_refinement();

  void init_border_nodes();

  template <typename Lambda>
  NodeID poll_border_nodes(const NodeID count, Lambda &&lambda) {
    NodeID polled = 0;
    while (polled < count && _next_border_node < _border_nodes.size()) {
      const NodeID remaining = count - polled;
      const NodeID from = _next_border_node.fetch_add(remaining);
      const NodeID to =
          std::min<NodeID>(from + remaining, _border_nodes.size());

      for (NodeID current = from; current < to; ++current) {
        const NodeID node = _border_nodes[from];
        std::uint8_t free = 0;
        if (__atomic_compare_exchange_n(
                &_locked[node],
                &free,
                1,
                false,
                __ATOMIC_SEQ_CST,
                __ATOMIC_SEQ_CST
            )) {
          lambda(node);
          ++polled;
        }
      }
    }

    return polled;
  }

  bool has_border_nodes() const;

  bool lock_node(const NodeID u);
  void unlock_node(const NodeID u);

  PartitionedGraph *_p_graph;
  const PartitionContext *_p_ctx;
  const KwayFMRefinementContext *_fm_ctx;

  parallel::Atomic<NodeID> _next_border_node;
  tbb::concurrent_vector<NodeID> _border_nodes;
  NoinitVector<std::uint8_t> _locked;

  RecomputeGainCache _gain_cache;

  tbb::enumerable_thread_specific<BinaryMinHeap<EdgeWeight>> _pq_ets;
  tbb::enumerable_thread_specific<Marker<>> _marker_ets;
  NoinitVector<BlockID> _target_blocks;
};
} // namespace kaminpar::shm
