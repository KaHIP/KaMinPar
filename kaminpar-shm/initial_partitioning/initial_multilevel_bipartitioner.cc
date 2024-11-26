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

SET_DEBUG(true);

}

InitialMultilevelBipartitioner::InitialMultilevelBipartitioner(const Context &ctx)
    : _ctx(ctx),
      _i_ctx(ctx.initial_partitioning),
      _coarsener(std::make_unique<InitialCoarsener>(_i_ctx.coarsening)),
      _bipartitioner(std::make_unique<InitialPoolBipartitioner>(_i_ctx.pool)),
      _refiner(create_initial_refiner(_i_ctx.refinement)) {}

void InitialMultilevelBipartitioner::initialize(
    const CSRGraph &graph, const BlockID current_block, const BlockID current_k
) {
  KASSERT(graph.n() > 0u);
  _graph = &graph;

  const BlockID first_sub_block =
      partitioning::compute_first_sub_block(current_block, current_k, _ctx.partition.k);
  const BlockID first_invalid_sub_block =
      partitioning::compute_first_invalid_sub_block(current_block, current_k, _ctx.partition.k);
  const BlockID num_sub_blocks =
      partitioning::compute_final_k(current_block, current_k, _ctx.partition.k);
  const auto [num_sub_blocks_b0, num_sub_blocks_b1] = math::split_integral(num_sub_blocks);

  std::vector<BlockWeight> max_block_weights{
      _ctx.partition.total_max_block_weights(first_sub_block, first_sub_block + num_sub_blocks_b0),
      _ctx.partition.total_max_block_weights(
          first_sub_block + num_sub_blocks_b0, first_invalid_sub_block
      )
  };

  DBG << "For block " << current_block << " of " << current_k << ": spans sub-blocks ["
      << first_sub_block << ", " << first_invalid_sub_block << "), split weight "
      << _ctx.partition.total_max_block_weights(first_sub_block, first_invalid_sub_block)
      << " into " << max_block_weights[0] << " and " << max_block_weights[1];

  if (_i_ctx.use_adaptive_epsilon) {
    const double adaptive_epsilon = partitioning::compute_2way_adaptive_epsilon(
        graph.total_node_weight(), num_sub_blocks, _ctx.partition
    );
    for (BlockWeight &max_block_weight : max_block_weights) {
      max_block_weight *= (1.0 + adaptive_epsilon) / (1.0 + _ctx.partition.epsilon());
    }
    _p_ctx.setup(graph, std::move(max_block_weights), true);
  } else {
    _p_ctx.setup(graph, std::move(max_block_weights), true);
  }

  _coarsener->init(graph);
  _refiner->init(graph);

  const int num_bipartition_repetitions = std::ceil(
      _i_ctx.pool.repetition_multiplier * num_sub_blocks / math::ceil_log2(_ctx.partition.k)
  );
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
  // DBG << "Coarsen: n=" << c_graph->n() << " m=" << c_graph->m();
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

    // DBG << "-> "                                              //
    //<< "n=" << new_c_graph->n() << " "                    //
    //<< "m=" << new_c_graph->m() << " "                    //
    //<< "max_cluster_weight=" << max_cluster_weight << " " //
    //<< ((shrunk) ? "" : "==> terminate");                 //

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
  // DBG << "Uncoarsen: n=" << p_graph.n() << " m=" << p_graph.m();

  while (!_coarsener->empty()) {
    p_graph = _coarsener->uncoarsen(std::move(p_graph));

    _refiner->init(p_graph.graph());
    _refiner->refine(p_graph, _p_ctx);

    // DBG << "-> "                                                 //
    //<< "n=" << p_graph.n() << " "                            //
    //<< "m=" << p_graph.m() << " "                            //
    //<< "cut=" << metrics::edge_cut_seq(p_graph) << " "       //
    //<< "imbalance=" << metrics::imbalance(p_graph) << " "    //
    //<< "feasible=" << metrics::is_feasible(p_graph, _p_ctx); //
  }

  return p_graph;
}

} // namespace kaminpar::shm
