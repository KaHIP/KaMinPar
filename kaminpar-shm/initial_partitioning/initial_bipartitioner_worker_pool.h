/*******************************************************************************
 * Thread-local pool of initial bipartitioner workers.
 *
 * @file:   initial_worker_pool.h
 * @author: Daniel Seemaier
 * @date:   05.06.2024
 ******************************************************************************/
#pragma once

#include <vector>

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-shm/initial_partitioning/initial_multilevel_bipartitioner.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

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

class InitialBipartitionerWorkerPool {
  SET_DEBUG(false);

public:
  explicit InitialBipartitionerWorkerPool(const Context &ctx) : _ctx(ctx) {}

  PartitionedGraph bipartition(
      const Graph *graph,
      const BlockID current_block,
      const BlockID current_k,
      const bool partition_lifespan
  ) {
    const CSRGraph *csr = dynamic_cast<const CSRGraph *>(graph->underlying_graph());

    // If we work with something other than a CSRGraph, construct a CSR copy to call the initial
    // partitioning code. This is only necessary if the graph is too small for coarsening *and* we
    // are using graph compression.
    std::unique_ptr<CSRGraph> csr_copy;
    if (csr == nullptr) {
      DBG << "Bipartitioning a non-CSR graph is not supported by the initial partitioning code: "
             "constructing a CSR-graph copy of the given graph with n="
          << graph->n() << ", m=" << graph->m();
      csr_copy = std::make_unique<CSRGraph>(*graph);
      csr = csr_copy.get();
    }

    auto bipartition = [&] {
      if (graph->n() == 0) {
        return StaticArray<BlockID>{};
      }

      InitialMultilevelBipartitioner bipartitioner = get();
      bipartitioner.initialize(*graph, *csr, current_block, current_k);
      auto bipartition = bipartitioner.partition(nullptr).take_raw_partition();

      if (partition_lifespan) {
        StaticArray<BlockID> owned_bipartition(bipartition.size(), static_array::noinit);
        std::copy(bipartition.begin(), bipartition.end(), owned_bipartition.begin());

        put(std::move(bipartitioner));

        return owned_bipartition;
      } else {
        put(std::move(bipartitioner));
        return bipartition;
      }
    }();

    PartitionedGraph p_graph(PartitionedGraph::seq{}, *graph, 2, std::move(bipartition));
    return p_graph;
  }

  void free() {
    _pool_ets.clear();
  }

private:
  InitialMultilevelBipartitioner get() {
    auto &pool = _pool_ets.local();

    if (!pool.empty()) {
      auto initial_partitioner = std::move(pool.back());
      pool.pop_back();
      return initial_partitioner;
    }

    return InitialMultilevelBipartitioner(_ctx);
  }

  void put(InitialMultilevelBipartitioner initial_partitioner) {
    auto &pool = _pool_ets.local();
    pool.push_back(std::move(initial_partitioner));
  }

  const Context &_ctx;
  tbb::enumerable_thread_specific<std::vector<InitialMultilevelBipartitioner>> _pool_ets;
};

} // namespace kaminpar::shm
