/*******************************************************************************
 * Coarsener that is optimized to contract clusterings.
 *
 * @file:   basic_cluster_coarsener.cc
 * @author: Daniel Seemaier
 * @date:   29.09.2021
 ******************************************************************************/
#include "kaminpar-shm/coarsening/basic_cluster_coarsener.h"

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"

namespace kaminpar::shm {

namespace {

SET_DEBUG(false);

}

BasicClusterCoarsener::BasicClusterCoarsener(const Context &ctx, const PartitionContext &p_ctx)
    : AbstractClusterCoarsener(ctx, p_ctx) {}

bool BasicClusterCoarsener::coarsen() {
  START_HEAP_PROFILER("Allocation");
  RECORD("clustering") StaticArray<NodeID> clustering(current().n(), static_array::noinit);
  STOP_HEAP_PROFILER();

  const bool free_allocated_memory = !keep_allocated_memory();
  const NodeID prev_n = current().n();

  compute_clustering_for_current_graph(clustering);
  contract_current_graph_and_push(clustering);

  if (free_allocated_memory) {
    _contraction_m_ctx.buckets.free();
    _contraction_m_ctx.buckets_index.free();
    _contraction_m_ctx.all_buffered_nodes.free();
  }

  return has_not_converged(prev_n);
}

} // namespace kaminpar::shm
