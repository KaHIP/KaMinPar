/*******************************************************************************
 * @file:   initial_partitioning_facade.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Facade for sequential initial partitioning.
 ******************************************************************************/
#pragma once

#include <memory>

#include "kaminpar-shm/coarsening/max_cluster_weights.h"
#include "kaminpar-shm/initial_partitioning/initial_coarsener.h"
#include "kaminpar-shm/initial_partitioning/initial_refiner.h"
#include "kaminpar-shm/initial_partitioning/pool_bipartitioner.h"
#include "kaminpar-shm/partitioning/partition_utils.h"

#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
struct InitialPartitionerTimings {
  std::uint64_t coarsening_ms = 0;
  std::uint64_t coarsening_misc_ms = 0;
  std::uint64_t coarsening_call_ms = 0;
  std::uint64_t uncoarsening_ms = 0;
  std::uint64_t bipartitioning_ms = 0;
  std::uint64_t total_ms = 0;
  std::uint64_t misc_ms = 0;
  ip::InitialCoarsenerTimings coarsening{};

  InitialPartitionerTimings &operator+=(const InitialPartitionerTimings &other) {
    coarsening_ms += other.coarsening_ms;
    uncoarsening_ms += other.uncoarsening_ms;
    bipartitioning_ms += other.bipartitioning_ms;
    misc_ms += other.misc_ms;
    coarsening += other.coarsening;
    coarsening_misc_ms += other.coarsening_misc_ms;
    coarsening_call_ms += other.coarsening_call_ms;
    total_ms += other.total_ms;
    return *this;
  }
};

class InitialPartitioner {
  SET_DEBUG(false);

public:
  struct MemoryContext {
    MemoryContext() = default;

    MemoryContext(const MemoryContext &) = delete;
    MemoryContext &operator=(const MemoryContext &) = delete;

    MemoryContext(MemoryContext &&) noexcept = default;
    MemoryContext &operator=(MemoryContext &&) noexcept = default;

    ip::InitialCoarsener::MemoryContext coarsener_m_ctx;
    ip::InitialRefiner::MemoryContext refiner_m_ctx;
    ip::PoolBipartitioner::MemoryContext pool_m_ctx;

    [[nodiscard]] std::size_t memory_in_kb() const {
      return coarsener_m_ctx.memory_in_kb() + refiner_m_ctx.memory_in_kb() +
             pool_m_ctx.memory_in_kb();
    }
  };

  InitialPartitioner(
      const CSRGraph &graph, const Context &ctx, const BlockID final_k, MemoryContext m_ctx = {}
  )
      : _m_ctx(std::move(m_ctx)),
        _graph(graph),
        _i_ctx(ctx.initial_partitioning),
        _coarsener(&_graph, _i_ctx.coarsening, std::move(_m_ctx.coarsener_m_ctx)) {
    const auto [final_k1, final_k2] = math::split_integral(final_k);
    _p_ctx = create_bipartition_context(_graph, final_k1, final_k2, ctx.partition);
    DBG << " -> created _p_ctx with max weights: " << _p_ctx.block_weights.max(0) << " + "
        << _p_ctx.block_weights.max(1);

    _refiner =
        create_initial_refiner(_graph, _p_ctx, _i_ctx.refinement, std::move(_m_ctx.refiner_m_ctx));

    // O(R * k) initial bisections -> O(n + R * C * k) for the whole graphutils
    _num_bipartition_repetitions =
        std::ceil(_i_ctx.repetition_multiplier * final_k / math::ceil_log2(ctx.partition.k));
  }

  MemoryContext free() {
    _m_ctx.refiner_m_ctx = _refiner->free();
    _m_ctx.coarsener_m_ctx = _coarsener.free();
    return std::move(_m_ctx);
  }

  PartitionedCSRGraph partition(InitialPartitionerTimings *timings = nullptr) {
    timer::LocalTimer timer_total;
    timer::LocalTimer timer_section;

    timer_total.reset();
    timer_section.reset();
    const CSRGraph *c_graph = coarsen(timings);
    if (timings) {
      timings->coarsening_ms += timer_section.elapsed();
    }

    timer_section.reset();
    ip::PoolBipartitionerFactory factory;
    auto bipartitioner = factory.create(*c_graph, _p_ctx, _i_ctx, std::move(_m_ctx.pool_m_ctx));
    bipartitioner->set_num_repetitions(_num_bipartition_repetitions);
    if (timings) {
      timings->misc_ms += timer_section.elapsed();
    }

    timer_section.reset();
    PartitionedCSRGraph p_graph = bipartitioner->bipartition();
    if (timings) {
      timings->bipartitioning_ms += timer_section.elapsed();
    }

    timer_section.reset();
    _m_ctx.pool_m_ctx = bipartitioner->free();
    if (timings) {
      //timings->misc_ms += timer_section.elapsed();
    }

    timer_section.reset();
    p_graph = uncoarsen(std::move(p_graph));
    if (timings) {
      timings->uncoarsening_ms += timer_section.elapsed();
      timings->total_ms += timer_total.elapsed();
    }

    return p_graph;
  }

private:
  const CSRGraph *coarsen(InitialPartitionerTimings *timings) {
    timer::LocalTimer timer;

    timer.reset();
    const InitialCoarseningContext &c_ctx = _i_ctx.coarsening;
    const NodeWeight max_cluster_weight = compute_max_cluster_weight<NodeWeight>(
        _i_ctx.coarsening, _p_ctx, _graph.n(), _graph.total_node_weight()
    );

    const CSRGraph *c_graph = &_graph;
    bool shrunk = true;
    DBG << "Coarsen: n=" << c_graph->n() << " m=" << c_graph->m();
    if (timings) {
      timings->coarsening_misc_ms += timer.elapsed();
    }

    while (shrunk && c_graph->n() > c_ctx.contraction_limit) {
      timer.reset();
      auto new_c_graph = _coarsener.coarsen(STATIC_MAX_CLUSTER_WEIGHT(max_cluster_weight));
      if (timings) {
        timings->coarsening_call_ms += timer.elapsed();
      }

      shrunk = new_c_graph != c_graph;

      DBG << "-> "                                              //
          << "n=" << new_c_graph->n() << " "                    //
          << "m=" << new_c_graph->m() << " "                    //
          << "max_cluster_weight=" << max_cluster_weight << " " //
          << ((shrunk) ? "" : "==> terminate");                 //

      if (shrunk) {
        c_graph = new_c_graph;
      }
    }

    if (timings) {
      timings->coarsening += _coarsener.timings();
    }

    return c_graph;
  }

  PartitionedCSRGraph uncoarsen(PartitionedCSRGraph p_graph) {
    DBG << "Uncoarsen: n=" << p_graph.n() << " m=" << p_graph.m();

    while (!_coarsener.empty()) {
      p_graph = _coarsener.uncoarsen(std::move(p_graph));
      _refiner->initialize(p_graph.graph());
      _refiner->refine(p_graph, _p_ctx);

      DBG << "-> "                                                 //
          << "n=" << p_graph.n() << " "                            //
          << "m=" << p_graph.m() << " "                            //
          << "cut=" << metrics::edge_cut_seq(p_graph) << " "       //
          << "imbalance=" << metrics::imbalance(p_graph) << " "    //
          << "feasible=" << metrics::is_feasible(p_graph, _p_ctx); //
    }

    return p_graph;
  }

  MemoryContext _m_ctx;
  const CSRGraph &_graph;
  const InitialPartitioningContext &_i_ctx;
  PartitionContext _p_ctx;
  ip::InitialCoarsener _coarsener;
  std::unique_ptr<ip::InitialRefiner> _refiner;

  std::size_t _num_bipartition_repetitions;
};
} // namespace kaminpar::shm
