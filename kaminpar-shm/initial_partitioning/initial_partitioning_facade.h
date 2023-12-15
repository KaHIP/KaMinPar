/*******************************************************************************
 * @file:   initial_partitioning_facade.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Facade for sequential initial partitioning.
 ******************************************************************************/
#pragma once

#include <memory>

#include "kaminpar-shm/initial_partitioning/initial_coarsener.h"
#include "kaminpar-shm/initial_partitioning/initial_refiner.h"
#include "kaminpar-shm/initial_partitioning/pool_bipartitioner.h"
#include "kaminpar-shm/partition_utils.h"

namespace kaminpar::shm {
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
      const Graph &graph, const Context &ctx, const BlockID final_k, MemoryContext m_ctx = {}
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

  PartitionedGraph partition() {
    const Graph *c_graph = coarsen();

    DBG << "Calling bipartitioner on coarsest graph with n=" << c_graph->n()
        << " m=" << c_graph->m();
    ip::PoolBipartitionerFactory factory;
    auto bipartitioner = factory.create(*c_graph, _p_ctx, _i_ctx, std::move(_m_ctx.pool_m_ctx));
    bipartitioner->set_num_repetitions(_num_bipartition_repetitions);
    PartitionedGraph p_graph = bipartitioner->bipartition();
    _m_ctx.pool_m_ctx = bipartitioner->free();

    DBG << "Bipartitioner result: "                              //
        << "cut=" << metrics::edge_cut_seq(p_graph) << " " //
        << "imbalance=" << metrics::imbalance(p_graph) << " "    //
        << "feasible=" << metrics::is_feasible(p_graph, _p_ctx); //

    return uncoarsen(std::move(p_graph));
  }

private:
  const Graph *coarsen() {
    const InitialCoarseningContext &c_ctx = _i_ctx.coarsening;
    const NodeWeight max_cluster_weight =
        compute_max_cluster_weight(_i_ctx.coarsening, _graph, _p_ctx);

    const Graph *c_graph = &_graph;
    bool shrunk = true;
    DBG << "Coarsen: n=" << c_graph->n() << " m=" << c_graph->m();

    while (shrunk && c_graph->n() > c_ctx.contraction_limit) {
      auto new_c_graph = _coarsener.coarsen(STATIC_MAX_CLUSTER_WEIGHT(max_cluster_weight));
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

    return c_graph;
  }

  PartitionedGraph uncoarsen(PartitionedGraph p_graph) {
    DBG << "Uncoarsen: n=" << p_graph.n() << " m=" << p_graph.m();

    while (!_coarsener.empty()) {
      p_graph = _coarsener.uncoarsen(std::move(p_graph));
      _refiner->initialize(p_graph.graph());
      _refiner->refine(p_graph, _p_ctx);

      DBG << "-> "                                                 //
          << "n=" << p_graph.n() << " "                            //
          << "m=" << p_graph.m() << " "                            //
          << "cut=" << metrics::edge_cut_seq(p_graph) << " " //
          << "imbalance=" << metrics::imbalance(p_graph) << " "    //
          << "feasible=" << metrics::is_feasible(p_graph, _p_ctx); //
    }

    return p_graph;
  }

  MemoryContext _m_ctx;
  const Graph &_graph;
  const InitialPartitioningContext &_i_ctx;
  PartitionContext _p_ctx;
  ip::InitialCoarsener _coarsener;
  std::unique_ptr<ip::InitialRefiner> _refiner;

  std::size_t _num_bipartition_repetitions;
};
} // namespace kaminpar::shm
