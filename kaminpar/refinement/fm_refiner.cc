/*******************************************************************************
 * @file:   fm_refiner.cc
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 * @brief:  Parallel k-way FM refinement algorithm.
 ******************************************************************************/
#include "kaminpar/refinement/fm_refiner.h"

#include <cmath>

#include <tbb/concurrent_vector.h>
#include <tbb/task_arena.h>

#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/datastructures/partitioned_graph.h"
#include "kaminpar/definitions.h"
#include "kaminpar/refinement/stopping_policies.h"

#include "common/datastructures/marker.h"
#include "common/parallel/atomic.h"

namespace kaminpar::shm {
namespace {
struct Move {
  NodeID node;
  BlockID from;
  BlockID to;
};
} // namespace

FMRefiner::FMRefiner(const Context &ctx)
    : _fm_ctx(ctx.refinement.kway_fm),
      _locked(ctx.partition.n),
      _gain_cache(ctx.partition.k, ctx.partition.n),
      _shared_pq_handles(ctx.partition.n),
      _target_blocks(ctx.partition.n) {
  tbb::parallel_for<std::size_t>(
      0,
      _shared_pq_handles.size(),
      [&](std::size_t i) {
        _shared_pq_handles[i] = SharedBinaryMaxHeap<EdgeWeight>::kInvalidID;
      }
  );
}

void FMRefiner::initialize(const PartitionedGraph &p_graph) {
  _gain_cache.initialize(*_p_graph);
}

bool FMRefiner::refine(
    PartitionedGraph &p_graph, const PartitionContext &p_ctx
) {
  _p_graph = &p_graph;
  _p_ctx = &p_ctx;

  // Reset refiner state
  p_graph.pfor_nodes([&](NodeID u) { _locked[u] = 0; });
  init_border_nodes();
  _expected_total_gain = 0;

  // Start one worker per thread
  std::atomic<int> next_id = 0;
  tbb::parallel_for<int>(0, tbb::this_task_arena::max_concurrency(), [&](int) {
    LocalizedFMRefiner localized_fm(++next_id, p_ctx, _fm_ctx, p_graph, *this);

    // Workers try to pull seed nodes from the remaining border nodes, until
    // there are no more border nodes left
    while (has_border_nodes()) {
      _expected_total_gain += localized_fm.run();
    }
  });

  return _expected_total_gain;
}

void FMRefiner::init_border_nodes() {
  _border_nodes.clear();
  _p_graph->pfor_nodes([&](const NodeID u) {
    if (_gain_cache.is_border_node(u, _p_graph->block(u))) {
      _border_nodes.push_back(u);
    }
    _locked[u] = 0;
  });
}

bool FMRefiner::has_border_nodes() const {
  return _next_border_node < _border_nodes.size();
}

bool FMRefiner::lock_node(const NodeID u, const int id) {
  std::uint8_t free = 0;
  return __atomic_compare_exchange_n(
      &_locked[u], &free, id, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
  );
}

int FMRefiner::owner(const NodeID u) {
  return _locked[u];
}

void FMRefiner::unlock_node(const NodeID u) {
  __atomic_store_n(&_locked[u], 0, __ATOMIC_RELAXED);
  if (_gain_cache.is_border_node(u, _p_graph->block(u))) {
    _border_nodes.push_back(u);
  }
}
} // namespace kaminpar::shm
