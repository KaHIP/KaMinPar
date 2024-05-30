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
  InitialPartitioner(const Context &ctx)
      : _ctx(ctx),
        _i_ctx(ctx.initial_partitioning),
        _coarsener(std::make_unique<ip::InitialCoarsener>(_i_ctx.coarsening)),
        _bipartitioner(std::make_unique<ip::PoolBipartitioner>(_i_ctx)),
        _refiner(ip::create_initial_refiner(_i_ctx.refinement)) {}

  void init(const CSRGraph &graph, const BlockID final_k) {
    const auto [final_k1, final_k2] = math::split_integral(final_k);

    _p_ctx = create_bipartition_context(*_graph, final_k1, final_k2, _ctx.partition);
    _graph = &graph;

    _coarsener->init(graph);
    _refiner->init(graph);

    const std::size_t num_bipartition_repetitions =
        std::ceil(_i_ctx.repetition_multiplier * final_k / math::ceil_log2(_ctx.partition.k));
    _bipartitioner->set_num_repetitions(num_bipartition_repetitions);
  }

  PartitionedCSRGraph partition(InitialPartitionerTimings *timings = nullptr) {
    timer::LocalTimer timer;

    timer.reset();
    const CSRGraph *c_graph = coarsen(timings);
    if (timings) {
      timings->coarsening_ms += timer.elapsed();
    }

    timer.reset();
    _bipartitioner->init(*c_graph, _p_ctx);
    PartitionedCSRGraph p_graph = _bipartitioner->bipartition();
    if (timings) {
      timings->bipartitioning_ms += timer.elapsed();
    }

    timer.reset();
    p_graph = uncoarsen(std::move(p_graph));
    if (timings) {
      timings->uncoarsening_ms += timer.elapsed();
    }

    return p_graph;
  }

private:
  const CSRGraph *coarsen(InitialPartitionerTimings *timings) {
    timer::LocalTimer timer;

    timer.reset();
    const InitialCoarseningContext &c_ctx = _i_ctx.coarsening;
    const NodeWeight max_cluster_weight = compute_max_cluster_weight<NodeWeight>(
        _i_ctx.coarsening, _p_ctx, _graph->n(), _graph->total_node_weight()
    );

    const CSRGraph *c_graph = _graph;

    bool shrunk = true;
    DBG << "Coarsen: n=" << c_graph->n() << " m=" << c_graph->m();
    if (timings) {
      timings->coarsening_misc_ms += timer.elapsed();
    }

    while (shrunk && c_graph->n() > c_ctx.contraction_limit) {
      timer.reset();
      auto new_c_graph = _coarsener->coarsen(max_cluster_weight);
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
      timings->coarsening += _coarsener->timings();
    }

    return c_graph;
  }

  PartitionedCSRGraph uncoarsen(PartitionedCSRGraph p_graph) {
    DBG << "Uncoarsen: n=" << p_graph.n() << " m=" << p_graph.m();

    while (!_coarsener->empty()) {
      p_graph = _coarsener->uncoarsen(std::move(p_graph));

      _refiner->init(p_graph.graph());
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

  const CSRGraph *_graph;
  PartitionContext _p_ctx;

  const Context &_ctx;
  const InitialPartitioningContext &_i_ctx;

  std::unique_ptr<ip::InitialCoarsener> _coarsener;
  std::unique_ptr<ip::PoolBipartitioner> _bipartitioner;
  std::unique_ptr<ip::InitialRefiner> _refiner;
};
} // namespace kaminpar::shm
