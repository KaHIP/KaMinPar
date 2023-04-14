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
#include "kaminpar/metrics.h"
#include "kaminpar/refinement/stopping_policies.h"

#include "common/datastructures/marker.h"
#include "common/parallel/atomic.h"
#include "common/timer.h"

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
  // Initialize shared PQ handles -- @todo clean this up
  tbb::parallel_for<std::size_t>(
      0,
      _shared_pq_handles.size(),
      [&](std::size_t i) {
        _shared_pq_handles[i] = SharedBinaryMaxHeap<EdgeWeight>::kInvalidID;
      }
  );
}

bool FMRefiner::refine(
    PartitionedGraph &p_graph, const PartitionContext &p_ctx
) {
  _p_graph = &p_graph;
  _p_ctx = &p_ctx;

  SCOPED_TIMER("FM");

  START_TIMER("Initialize gain cache");
  _gain_cache.initialize(*_p_graph);
  STOP_TIMER();

  const EdgeWeight initial_cut = metrics::edge_cut(*_p_graph);

  // Total gain across all iterations
  // This value is not accurate, but the sum of all gains achieved on local
  // delta graphs
  EdgeWeight total_expected_gain = 0;

  for (int iteration = 0; iteration < _fm_ctx.num_iterations; ++iteration) {
    SCOPED_TIMER("Iteration", std::to_string(iteration));

    // Gains of the current iterations
    tbb::enumerable_thread_specific<EdgeWeight> expected_gain_ets;

    // Make sure that we work with correct gains
    KASSERT(
        _gain_cache.validate(*_p_graph),
        "gain cache invalid before iteration " << iteration,
        assert::heavy
    );

    // Find current border nodes
    // This also initializes _locked[], or resets it after the first round
    init_border_nodes();

    LOG << "Starting FM iteration " << iteration << " with "
        << _border_nodes.size() << " border nodes and "
        << tbb::this_task_arena::max_concurrency() << " worker threads";

    // Start one worker per thread
    std::atomic<int> next_id = 0;
    tbb::parallel_for<int>(
        0,
        tbb::this_task_arena::max_concurrency(),
        [&](int) {
          EdgeWeight &expected_gain = expected_gain_ets.local();
          LocalizedFMRefiner localized_fm(
              ++next_id, p_ctx, _fm_ctx, p_graph, *this
          );

          // The workers attempt to extract seed nodes from the border nodes
          // that are still available, continuing this process until there are
          // no more border nodes
          while (has_border_nodes()) {
            expected_gain += localized_fm.run();
          }
        }
    );

    // Abort early if the expected cut improvement falls below the abortion
    // threshold
    // @todo is it feasible to work with these "expected" values? Do we need
    // accurate gains?
    const EdgeWeight expected_gain = expected_gain_ets.combine(std::plus{});
    total_expected_gain += expected_gain;
    const EdgeWeight expected_current_cut = initial_cut - total_expected_gain;
    const EdgeWeight abortion_threshold =
        expected_current_cut * _fm_ctx.improvement_abortion_threshold;
    LOG << "Expected total gain after iteration " << iteration << ": "
        << total_expected_gain
        << ", actual gain: " << initial_cut - metrics::edge_cut(*_p_graph);

    if (expected_gain < abortion_threshold) {
      DBG << "Aborting because expected gain is below threshold "
          << abortion_threshold;
      break;
    }
  }

  return total_expected_gain;
}

void FMRefiner::init_border_nodes() {
  _border_nodes.clear();
  _p_graph->pfor_nodes([&](const NodeID u) {
    if (_gain_cache.is_border_node(u, _p_graph->block(u))) {
      _border_nodes.push_back(u);
    }
    _locked[u] = 0;
  });
  _next_border_node = 0;
}

bool FMRefiner::has_border_nodes() const {
  return _next_border_node < _border_nodes.size();
}

bool FMRefiner::lock_node(const NodeID u, const int id) {
  int free = 0;
  return __atomic_compare_exchange_n(
      &_locked[u], &free, id, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST
  );
}

int FMRefiner::owner(const NodeID u) {
  return _locked[u];
}

void FMRefiner::unlock_node(const NodeID u) {
  __atomic_store_n(&_locked[u], 0, __ATOMIC_RELAXED);
  //if (_gain_cache.is_border_node(u, _p_graph->block(u))) {
  //  _border_nodes.push_back(u);
  //}
}
} // namespace kaminpar::shm
