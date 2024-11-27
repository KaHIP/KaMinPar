/*******************************************************************************
 * Utility functions for common operations used by partitioning schemes.
 *
 * @file:   helper.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-shm/graphutils/subgraph_extractor.h"
#include "kaminpar-shm/initial_partitioning/initial_bipartitioner_worker_pool.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/metrics.h"

#include "kaminpar-common/assert.h"

namespace kaminpar::shm::partitioning {

PartitionContext create_kway_context(const Context &input_ctx, const PartitionedGraph &p_graph);

using SubgraphMemoryEts = tbb::enumerable_thread_specific<graph::SubgraphMemory>;
using TemporarySubgraphMemoryEts = tbb::enumerable_thread_specific<graph::TemporarySubgraphMemory>;

/**
 * Peforms recursive bipartitioning on the blocks of `p_graph` to obtain a partition with
 * `desired_k` blocks.
 *
 * In contrast to the non-lazy version, this function does not extract all block-induced subgraphs
 * of `p_graph` in advance. Instead, it extracts the blocks one-by-one and immediately partitions
 * them.
 *
 * @param p_graph The partitioned graph of which the blocks will be recursively bipartitioned.
 * @param desired_k The number of blocks in the final partition.
 * @param input_ctx The input context, used to compute max block weights.
 * @param extraction_mem_pool_ets ...
 * @param tmp_extraction_mem_pool_ets ...
 * @param bipartitioner_pool The worker pool used to compute the bipartitions.
 * @param num_active_threads The number of currently active threads (in this replication branch of
 * deep multilevel).
 */
void extend_partition_lazy_extraction(
    PartitionedGraph &p_graph,
    BlockID desired_k,
    const Context &input_ctx,
    SubgraphMemoryEts &extraction_mem_pool_ets,
    TemporarySubgraphMemoryEts &tmp_extraction_mem_pool_ets,
    InitialBipartitionerWorkerPool &bipartitioner_pool,
    int num_active_threads
);

/**
 * @deprecated Use `extend_partition_lazy_extraction` instead.
 */
void extend_partition(
    PartitionedGraph &p_graph,
    BlockID desired_k,
    const Context &input_ctx,
    graph::SubgraphMemory &subgraph_memory,
    TemporarySubgraphMemoryEts &tmp_extraction_mem_pool_ets,
    InitialBipartitionerWorkerPool &bipartitioner_pool,
    int num_active_threads
);

/**
 * @deprecated Use `extend_partition_lazy_extraction` instead.
 */
void extend_partition(
    PartitionedGraph &p_graph,
    BlockID desired_k,
    const Context &input_ctx,
    TemporarySubgraphMemoryEts &tmp_extraction_mem_pool_ets,
    InitialBipartitionerWorkerPool &bipartitioner_pool,
    int num_active_threads
);

void complete_partial_extend_partition(
    PartitionedGraph &p_graph,
    const Context &input_ctx,
    SubgraphMemoryEts &extraction_mem_pool_ets,
    TemporarySubgraphMemoryEts &tmp_extraction_mem_pool_ets,
    InitialBipartitionerWorkerPool &bipartitioner_pool
);

template <typename Iterator>
std::size_t select_best(
    const Iterator p_graphs_begin, const Iterator p_graphs_end, const PartitionContext &p_ctx
) {
  KASSERT(p_graphs_begin < p_graphs_end, "cannot select the best partition from an empty range");

  std::size_t best_index = 0;
  std::size_t current_index = 0;
  EdgeWeight best_cut = std::numeric_limits<EdgeWeight>::max();
  bool best_feasible = false;

  for (auto it = p_graphs_begin; it != p_graphs_end; ++it) {
    const auto &result = *it;
    const bool current_feasible = metrics::is_feasible(result, p_ctx);
    const EdgeWeight current_cut = metrics::edge_cut(result);

    if ((current_feasible == best_feasible && current_cut < best_cut) ||
        current_feasible > best_feasible) {
      best_index = current_index;
      best_cut = current_cut;
      best_feasible = current_feasible;
    }

    ++current_index;
  }

  return best_index;
}

inline std::size_t
select_best(const ScalableVector<PartitionedGraph> &p_graphs, const PartitionContext &p_ctx) {
  return select_best(p_graphs.begin(), p_graphs.end(), p_ctx);
}

} // namespace kaminpar::shm::partitioning
