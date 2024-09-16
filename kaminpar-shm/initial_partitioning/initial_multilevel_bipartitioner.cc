/*******************************************************************************
 * Simple interface for the initial (bi)partitioning code.
 *
 * Initial bipartitioning uses a sequential multilevel to compute high-quality
 * bipartitions.
 *
 * For coarsening, the code uses sequential label propagation, which is
 * interleaved with the construction of the next coarse graph. Bipartitioning
 * is done by a pool of simple algorithms (BFS, greedy graph growing, random).
 * Refinement is done by a 2-way sequential FM algorithm.
 *
 * Constructing an object of InitialPartitioner is relatively expensive;
 * especially if one wants to compute *many* bipartitions (i.e., if k is large).
 * Thus, objects should be kept in (thread-local!) memory and be re-used to
 * compute multiple bipartitions (call init() for each new graph).
 *
 * Data structures are re-allocated to a larger size whenever necessary and never
 * shrink.
 *
 * @file:   initial_multilevel_bipartitioner.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/initial_partitioning/initial_multilevel_bipartitioner.h"

#include "kaminpar-shm/coarsening/max_cluster_weights.h"
#include "kaminpar-shm/initial_partitioning/initial_coarsener.h"
#include "kaminpar-shm/initial_partitioning/initial_pool_bipartitioner.h"
#include "kaminpar-shm/initial_partitioning/initial_refiner.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/metrics.h"
#include "kaminpar-shm/partitioning/partition_utils.h"

#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {
namespace {
SET_DEBUG(false);
}

InitialMultilevelBipartitioner::InitialMultilevelBipartitioner(const Context &ctx)
    : _ctx(ctx),
      _i_ctx(ctx.initial_partitioning),
      _coarsener(std::make_unique<InitialCoarsener>(_i_ctx.coarsening)),
      _bipartitioner(std::make_unique<InitialPoolBipartitioner>(_i_ctx.pool)),
      _refiner(create_initial_refiner(_i_ctx.refinement)) {}

void InitialMultilevelBipartitioner::initialize(const CSRGraph &graph, const BlockID final_k) {
  KASSERT(final_k > 0u);
  KASSERT(graph.n() > 0u);

  _graph = &graph;

  const auto [final_k1, final_k2] = math::split_integral(final_k);
  _p_ctx = partitioning::create_bipartition_context(graph, final_k1, final_k2, _ctx.partition);

  _coarsener->init(graph);
  _refiner->init(graph);

  const std::size_t num_bipartition_repetitions =
      std::ceil(_i_ctx.pool.repetition_multiplier * final_k / math::ceil_log2(_ctx.partition.k));
  _bipartitioner->set_num_repetitions(num_bipartition_repetitions);
}

PartitionedCSRGraph InitialMultilevelBipartitioner::partition(InitialPartitionerTimings *timings) {
  timer::LocalTimer timer;

  timer.reset();
  const CSRGraph *c_graph = coarsen(timings);
  if (timings) {
    timings->coarsening_ms += timer.elapsed();
  }

  timer.reset();
  _bipartitioner->init(*c_graph, _p_ctx);
  PartitionedCSRGraph p_graph = _bipartitioner->bipartition();

  if (_i_ctx.refine_pool_partition) {
    _refiner->init(p_graph.graph());
    _refiner->refine(p_graph, _p_ctx);
  }

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

const CSRGraph *InitialMultilevelBipartitioner::coarsen(InitialPartitionerTimings *timings) {
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

PartitionedCSRGraph InitialMultilevelBipartitioner::uncoarsen(PartitionedCSRGraph p_graph) {
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
} // namespace kaminpar::shm
