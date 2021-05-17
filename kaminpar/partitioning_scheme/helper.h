/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#pragma once

#include "coarsening/i_coarsener.h"
#include "datastructure/graph.h"
#include "initial_partitioning/initial_partitioning_facade.h"
#include "refinement/i_balancer.h"
#include "refinement/i_refiner.h"
#include "definitions.h"

#include <tbb/concurrent_vector.h>

namespace kaminpar::partitioning {
struct InitialPartitionerMemoryPool {
  std::vector<ip::InitialPartitioner::MemoryContext> pool;

  ip::InitialPartitioner::MemoryContext get() {
    if (!pool.empty()) {
      auto m_ctx = std::move(pool.back());
      pool.pop_back();
      return m_ctx;
    }

    return {};
  }

  std::size_t memory_in_kb() const {
    std::size_t memory = 0;
    for (const auto &obj : pool) { memory += obj.memory_in_kb(); }
    return memory;
  }

  void put(ip::InitialPartitioner::MemoryContext m_ctx) { pool.push_back(std::move(m_ctx)); }
};

using GlobalInitialPartitionerMemoryPool = tbb::enumerable_thread_specific<InitialPartitionerMemoryPool>;
using TemporaryGraphExtractionBufferPool = tbb::enumerable_thread_specific<TemporarySubgraphMemory>;

namespace helper {
void update_partition_context(PartitionContext &p_ctx, const PartitionedGraph &p_graph);

PartitionedGraph uncoarsen_once(Coarsener *coarsener, PartitionedGraph p_graph, PartitionContext &current_p_ctx);

void refine(Refiner *refiner, Balancer *balancer, PartitionedGraph &p_graph, const PartitionContext &current_p_ctx,
            const RefinementContext &r_ctx);

PartitionedGraph bipartition(const Graph *graph, const BlockID final_k, const Context &input_ctx,
                             GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool);

void extend_partition_recursive(const Graph &graph, StaticArray<BlockID> &partition, const BlockID b0, const BlockID k,
                                const BlockID final_k, const Context &input_ctx, SubgraphMemory &subgraph_memory,
                                const SubgraphMemoryStartPosition position,
                                TemporaryGraphExtractionBufferPool &extraction_pool,
                                GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool);

void extend_partition(PartitionedGraph &p_graph, const BlockID k_prime, const Context &input_ctx,
                      PartitionContext &current_p_ctx, GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool);

void extend_partition(PartitionedGraph &p_graph, const BlockID k_prime, const Context &input_ctx,
                      PartitionContext &current_p_ctx, SubgraphMemory &subgraph_memory,
                      TemporaryGraphExtractionBufferPool &extraction_pool,
                      GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool);

void extend_partition(PartitionedGraph &p_graph, const BlockID k_prime, const Context &input_ctx,
                      PartitionContext &current_p_ctx, TemporaryGraphExtractionBufferPool &extraction_pool,
                      GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool);

bool coarsen_once(Coarsener *coarsener, const Graph *graph, const Context &input_ctx, PartitionContext &current_p_ctx);

// compute smallest k_prime such that it is a power of 2 and n / k_prime <= C
BlockID compute_k_for_n(const NodeID n, const Context &input_ctx);

std::size_t compute_num_copies(const Context &input_ctx, const NodeID n, const bool converged,
                               const std::size_t num_threads);

std::size_t select_best(const scalable_vector<PartitionedGraph> &p_graphs, const PartitionContext &p_ctx);

std::size_t select_best(const auto p_graphs_begin, const auto p_graphs_end, const PartitionContext &p_ctx) {
  SET_DEBUG(false);

  ASSERT(p_graphs_begin < p_graphs_end) << "cannot select best result from an empty range";
  DBG << "Select best result from " << std::distance(p_graphs_begin, p_graphs_end) << " " << (*p_graphs_begin).k()
      << "-way partitions";

  std::size_t best_index = 0;
  std::size_t current_index = 0;
  EdgeWeight best_cut = std::numeric_limits<EdgeWeight>::max();
  bool best_feasible = false;

  for (auto it = p_graphs_begin; it != p_graphs_end; ++it) {
    const auto &result = *it;
    const bool current_feasible = metrics::is_feasible(result, p_ctx);
    const EdgeWeight current_cut = metrics::edge_cut(result);

    if ((current_feasible == best_feasible && current_cut < best_cut) || current_feasible > best_feasible) {
      best_index = current_index;
      best_cut = current_cut;
      best_feasible = current_feasible;
    }

    ++current_index;
  }

  return best_index;
}

std::size_t compute_num_threads_for_parallel_ip(const Context &input_ctx);
} // namespace helper
} // namespace kaminpar::partitioning