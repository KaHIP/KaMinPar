/*******************************************************************************
 * Pseudo-refiner that calls Mt-KaHyPar.
 *
 * @file:   mtkahypar_refiner.cc
 * @author: Daniel Seemaier
 * @date:   01.07.2023
 ******************************************************************************/
#include "kaminpar-shm/refinement/adapters/mtkahypar_refiner.h"

#include "kaminpar-shm/context.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/metrics.h"

#include "kaminpar-common/datastructures/noinit_vector.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/random.h"

#ifdef KAMINPAR_HAVE_MTKAHYPAR_LIB
#include <libmtkahypar.h>
#endif // KAMINPAR_HAVE_MTKAHYPAR_LIB

namespace kaminpar::shm {
SET_DEBUG(true);

MtKaHyParRefiner::MtKaHyParRefiner(const Context &ctx) : _ctx(ctx) {}

bool MtKaHyParRefiner::refine(
    [[maybe_unused]] PartitionedGraph &p_graph, [[maybe_unused]] const PartitionContext &p_ctx
) {
#ifdef KAMINPAR_HAVE_MTKAHYPAR_LIB
  mt_kahypar_context_t *mt_kahypar_ctx = mt_kahypar_context_new();
  if (_ctx.refinement.mtkahypar.config_filename.empty()) {
    const bool toplevel = (p_graph.n() == _ctx.partition.n);

    if (toplevel && !_ctx.refinement.mtkahypar.fine_config_filename.empty()) {
      mt_kahypar_configure_context_from_file(
          mt_kahypar_ctx, _ctx.refinement.mtkahypar.fine_config_filename.c_str()
      );
    } else if (!toplevel && !_ctx.refinement.mtkahypar.coarse_config_filename.empty()) {
      mt_kahypar_configure_context_from_file(
          mt_kahypar_ctx, _ctx.refinement.mtkahypar.coarse_config_filename.c_str()
      );
    } else {
      mt_kahypar_load_preset(mt_kahypar_ctx, DEFAULT);
    }
  } else {
    mt_kahypar_configure_context_from_file(
        mt_kahypar_ctx, _ctx.refinement.mtkahypar.config_filename.c_str()
    );
  }
  mt_kahypar_set_partitioning_parameters(
      mt_kahypar_ctx,
      static_cast<mt_kahypar_partition_id_t>(p_ctx.k),
      p_ctx.epsilon,
      KM1,
      Random::get_seed()
  );

  NoinitVector<mt_kahypar_hypernode_weight_t> block_weights(p_ctx.k);
  p_graph.pfor_blocks([&](const BlockID b) { block_weights[b] = p_ctx.block_weights.max(b); });
  mt_kahypar_set_individual_target_block_weights(
      mt_kahypar_ctx, static_cast<mt_kahypar_partition_id_t>(p_ctx.k), block_weights.data()
  );

  mt_kahypar_set_context_parameter(mt_kahypar_ctx, VERBOSE, "1");
  mt_kahypar_initialize_thread_pool(_ctx.parallel.num_threads, true);

  const mt_kahypar_hypernode_id_t num_vertices = p_graph.n();
  const mt_kahypar_hyperedge_id_t num_edges = p_graph.m() / 2; // Only need one direction

  NoinitVector<EdgeID> edge_position(2 * num_edges);
  p_graph.pfor_nodes([&](const NodeID u) {
    for (const auto [e, v] : p_graph.neighbors(u)) {
      edge_position[e] = u < v;
    }
  });
  parallel::prefix_sum(edge_position.begin(), edge_position.end(), edge_position.begin());

  NoinitVector<mt_kahypar_hypernode_id_t> edges(2 * num_edges);
  NoinitVector<mt_kahypar_hypernode_weight_t> edge_weights(num_edges);
  NoinitVector<mt_kahypar_hypernode_weight_t> vertex_weights(num_vertices);
  edges.reserve(2 * num_edges);
  edge_weights.reserve(num_edges);

  p_graph.pfor_nodes([&](const NodeID u) {
    vertex_weights[u] = static_cast<mt_kahypar_hypernode_weight_t>(p_graph.node_weight(u));

    for (const auto [e, v] : p_graph.neighbors(u)) {
      if (v < u) { // Only need edges in one direction
        continue;
      }

      EdgeID position = edge_position[e] - 1;
      edges[2 * position] = static_cast<mt_kahypar_hypernode_id_t>(u);
      edges[2 * position + 1] = static_cast<mt_kahypar_hypernode_id_t>(v);
      edge_weights[position] = static_cast<mt_kahypar_hypernode_weight_t>(p_graph.edge_weight(e));
    }
  });

  mt_kahypar_hypergraph_t mt_kahypar_graph = mt_kahypar_create_graph(
      DEFAULT, num_vertices, num_edges, edges.data(), edge_weights.data(), vertex_weights.data()
  );

  DBG << "Partition metrics before Mt-KaHyPar refinement: cut=" << metrics::edge_cut(p_graph)
      << " imbalance=" << metrics::imbalance(p_graph);

  NoinitVector<mt_kahypar_partition_id_t> partition(num_vertices);
  p_graph.pfor_nodes([&](const NodeID u) { partition[u] = p_graph.block(u); });

  mt_kahypar_partitioned_hypergraph_t mt_kahypar_partitioned_graph =
      mt_kahypar_create_partitioned_hypergraph(
          mt_kahypar_graph,
          DEFAULT,
          static_cast<mt_kahypar_partition_id_t>(p_ctx.k),
          partition.data()
      );

  // Run refinement
  mt_kahypar_improve_partition(mt_kahypar_partitioned_graph, mt_kahypar_ctx, 1);

  // Copy partition back to our graph
  NoinitVector<mt_kahypar_partition_id_t> improved_partition(num_vertices);
  mt_kahypar_get_partition(mt_kahypar_partitioned_graph, improved_partition.data());
  p_graph.pfor_nodes([&](const NodeID u) { p_graph.set_block(u, improved_partition[u]); });

  DBG << "Partition metrics after Mt-KaHyPar refinement: cut=" << metrics::edge_cut(p_graph)
      << " imbalance=" << metrics::imbalance(p_graph);

  // Free Mt-KaHyPar structs
  mt_kahypar_free_partitioned_hypergraph(mt_kahypar_partitioned_graph);
  mt_kahypar_free_hypergraph(mt_kahypar_graph);
  mt_kahypar_free_context(mt_kahypar_ctx);

  return false;
#else  // KAMINPAR_HAVE_MTKAHYPAR_LIB
  LOG_WARNING << "Mt-KaHyPar is not available; skipping refinement";
  return false;
#endif // KAMINPAR_HAVE_MTKAHYPAR_LIB
}
} // namespace kaminpar::shm
