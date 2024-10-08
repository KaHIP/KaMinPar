/*******************************************************************************
 * Utility functions for common operations used by partitioning schemes.
 *
 * @file:   helper.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#pragma once

#include <tbb/concurrent_vector.h>

#include "kaminpar-shm/coarsening/coarsener.h"
#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/graphutils/subgraph_extractor.h"
#include "kaminpar-shm/initial_partitioning/initial_bipartitioner_worker_pool.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/refinement/refiner.h"

#include "kaminpar-common/assert.h"

namespace kaminpar::shm::partitioning {
using SubgraphMemoryEts = tbb::enumerable_thread_specific<graph::SubgraphMemory>;
using TemporarySubgraphMemoryEts = tbb::enumerable_thread_specific<graph::TemporarySubgraphMemory>;

void update_partition_context(
    PartitionContext &p_ctx, const PartitionedGraph &p_graph, BlockID input_k
);

PartitionedGraph uncoarsen_once(
    Coarsener *coarsener,
    PartitionedGraph p_graph,
    PartitionContext &current_p_ctx,
    const PartitionContext &input_p_ctx
);

struct BipartitionTimingInfo {
  std::uint64_t bipartitioner_init_ms = 0;
  std::uint64_t bipartitioner_ms = 0;
  std::uint64_t graph_init_ms = 0;
  std::uint64_t extract_ms = 0;
  std::uint64_t copy_ms = 0;
  std::uint64_t misc_ms = 0;
  InitialPartitionerTimings ip_timings{};

  BipartitionTimingInfo &operator+=(const BipartitionTimingInfo &other) {
    bipartitioner_init_ms += other.bipartitioner_init_ms;
    bipartitioner_ms += other.bipartitioner_ms;
    graph_init_ms += other.graph_init_ms;
    extract_ms += other.extract_ms;
    copy_ms += other.copy_ms;
    misc_ms += other.misc_ms;
    ip_timings += other.ip_timings;
    return *this;
  }
};

PartitionedGraph bipartition(
    const Graph *graph,
    BlockID final_k,
    InitialBipartitionerWorkerPool &bipartitioner_pool_ets,
    bool partition_lifespan,
    BipartitionTimingInfo *timing_info = nullptr
);

void refine(Refiner *refiner, PartitionedGraph &p_graph, const PartitionContext &current_p_ctx);

void extend_partition_lazy_extraction(
    PartitionedGraph &p_graph,
    BlockID k_prime,
    const Context &input_ctx,
    PartitionContext &current_p_ctx,
    SubgraphMemoryEts &extraction_mem_pool_ets,
    TemporarySubgraphMemoryEts &tmp_extraction_mem_pool_ets,
    InitialBipartitionerWorkerPool &bipartitioner_pool,
    std::size_t num_active_threads
);

void extend_partition(
    PartitionedGraph &p_graph,
    BlockID k_prime,
    const Context &input_ctx,
    PartitionContext &current_p_ctx,
    graph::SubgraphMemory &subgraph_memory,
    TemporarySubgraphMemoryEts &tmp_extraction_mem_pool_ets,
    InitialBipartitionerWorkerPool &bipartitioner_pool,
    std::size_t num_active_threads
);

void extend_partition(
    PartitionedGraph &p_graph,
    BlockID k_prime,
    const Context &input_ctx,
    PartitionContext &current_p_ctx,
    TemporarySubgraphMemoryEts &tmp_extraction_mem_pool_ets,
    InitialBipartitionerWorkerPool &bipartitioner_pool,
    std::size_t num_active_threads
);

bool coarsen_once(Coarsener *coarsener, const Graph *graph, PartitionContext &current_p_ctx);

std::size_t
select_best(const ScalableVector<PartitionedGraph> &p_graphs, const PartitionContext &p_ctx);

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

inline bool parallel_ip_mode(const InitialPartitioningMode &mode) {
  return mode == InitialPartitioningMode::ASYNCHRONOUS_PARALLEL ||
         mode == InitialPartitioningMode::SYNCHRONOUS_PARALLEL;
}
} // namespace kaminpar::shm::partitioning
