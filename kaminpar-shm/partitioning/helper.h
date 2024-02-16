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
#include "kaminpar-shm/definitions.h"
#include "kaminpar-shm/graphutils/subgraph_extractor.h"
#include "kaminpar-shm/initial_partitioning/initial_partitioning_facade.h"

#include "kaminpar-common/assert.h"

namespace kaminpar::shm::partitioning {
struct InitialPartitionerMemoryPool {
  std::vector<InitialPartitioner::MemoryContext> pool;

  InitialPartitioner::MemoryContext get() {
    if (!pool.empty()) {
      auto m_ctx = std::move(pool.back());
      pool.pop_back();
      return m_ctx;
    }

    return {};
  }

  [[nodiscard]] std::size_t memory_in_kb() const {
    std::size_t memory = 0;
    for (const auto &obj : pool) {
      memory += obj.memory_in_kb();
    }
    return memory;
  }

  void put(InitialPartitioner::MemoryContext m_ctx) {
    pool.push_back(std::move(m_ctx));
  }
};

using GlobalInitialPartitionerMemoryPool =
    tbb::enumerable_thread_specific<InitialPartitionerMemoryPool>;
using TemporaryGraphExtractionBufferPool =
    tbb::enumerable_thread_specific<graph::TemporarySubgraphMemory>;

namespace helper {
void update_partition_context(
    PartitionContext &p_ctx, const PartitionedGraph &p_graph, BlockID input_k
);

PartitionedGraph uncoarsen_once(
    Coarsener *coarsener,
    PartitionedGraph p_graph,
    PartitionContext &current_p_ctx,
    const PartitionContext &input_p_ctx
);

PartitionedGraph bipartition(
    const Graph *graph,
    BlockID final_k,
    const Context &input_ctx,
    GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool
);

void refine(Refiner *refiner, PartitionedGraph &p_graph, const PartitionContext &current_p_ctx);

void extend_partition_recursive(
    const Graph &graph,
    StaticArray<BlockID> &partition,
    BlockID b0,
    BlockID k,
    BlockID final_k,
    const Context &input_ctx,
    graph::SubgraphMemory &subgraph_memory,
    graph::SubgraphMemoryStartPosition position,
    TemporaryGraphExtractionBufferPool &extraction_pool,
    GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool
);

void extend_partition(
    PartitionedGraph &p_graph,
    BlockID k_prime,
    const Context &input_ctx,
    PartitionContext &current_p_ctx,
    GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool
);

void extend_partition(
    PartitionedGraph &p_graph,
    BlockID k_prime,
    const Context &input_ctx,
    PartitionContext &current_p_ctx,
    graph::SubgraphMemory &subgraph_memory,
    TemporaryGraphExtractionBufferPool &extraction_pool,
    GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool
);

void extend_partition(
    PartitionedGraph &p_graph,
    BlockID k_prime,
    const Context &input_ctx,
    PartitionContext &current_p_ctx,
    TemporaryGraphExtractionBufferPool &extraction_pool,
    GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool
);

bool coarsen_once(
    Coarsener *coarsener,
    const Graph *graph,
    const Context &input_ctx,
    PartitionContext &current_p_ctx
);

// compute smallest k_prime such that it is a power of 2 and n / k_prime <= C
BlockID compute_k_for_n(NodeID n, const Context &input_ctx);

std::size_t
compute_num_copies(const Context &input_ctx, NodeID n, bool converged, std::size_t num_threads);

std::size_t
select_best(const scalable_vector<PartitionedGraph> &p_graphs, const PartitionContext &p_ctx);

template <typename Iterator>
std::size_t select_best(
    const Iterator p_graphs_begin, const Iterator p_graphs_end, const PartitionContext &p_ctx
) {
  SET_DEBUG(false);

  KASSERT(p_graphs_begin < p_graphs_end, "cannot select best result from an empty range");
  DBG << "Select best result from " << std::distance(p_graphs_begin, p_graphs_end) << " "
      << (*p_graphs_begin).k() << "-way partitions";

  std::size_t best_index = 0;
  std::size_t current_index = 0;
  EdgeWeight best_cut = std::numeric_limits<EdgeWeight>::max();
  bool best_feasible = false;

  for (auto it = p_graphs_begin; it != p_graphs_end; ++it) {
    const auto &result = *it;
    const bool current_feasible = metrics::is_feasible(result, p_ctx);
    const EdgeWeight current_cut = metrics::edge_cut_seq(result);

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

std::size_t compute_num_threads_for_parallel_ip(const Context &input_ctx);

inline bool parallel_ip_mode(const InitialPartitioningMode &mode) {
  return mode == InitialPartitioningMode::ASYNCHRONOUS_PARALLEL ||
         mode == InitialPartitioningMode::SYNCHRONOUS_PARALLEL;
}
} // namespace helper
} // namespace kaminpar::shm::partitioning
