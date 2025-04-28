/*******************************************************************************
 * Pseudo-refiner that calls Mt-KaHyPar.
 *
 * @file:   mtkahypar_refiner.cc
 * @author: Daniel Seemaier
 * @date:   01.07.2023
 ******************************************************************************/
#include "kaminpar-shm/refinement/adapters/mtkahypar_refiner.h"

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/logger.h"

#ifdef KAMINPAR_HAVE_MTKAHYPAR_LIB
#include <mtkahypar.h>

#include "kaminpar-shm/metrics.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"
#endif // KAMINPAR_HAVE_MTKAHYPAR_LIB

namespace kaminpar::shm {

namespace {
SET_DEBUG(false);
}

MtKaHyParRefiner::MtKaHyParRefiner(const Context &ctx) : _ctx(ctx) {}

std::string MtKaHyParRefiner::name() const {
  return "Mt-KaHyPar";
}

void MtKaHyParRefiner::initialize([[maybe_unused]] const PartitionedGraph &p_graph) {}

bool MtKaHyParRefiner::refine(
    [[maybe_unused]] PartitionedGraph &p_graph, [[maybe_unused]] const PartitionContext &p_ctx
) {
#ifdef KAMINPAR_HAVE_MTKAHYPAR_LIB
  SCOPED_TIMER("Mt-KaHyPar");
  SCOPED_HEAP_PROFILER("Mt-KaHyPar");

  mt_kahypar_error_t error{};

  mt_kahypar_context_t *mt_kahypar_ctx;
  if (_ctx.refinement.mtkahypar.config_filename.empty()) {
    const bool toplevel = (p_graph.n() == _ctx.partition.n);

    if (toplevel && !_ctx.refinement.mtkahypar.fine_config_filename.empty()) {
      mt_kahypar_ctx = mt_kahypar_context_from_file(
          _ctx.refinement.mtkahypar.fine_config_filename.c_str(), &error
      );
    } else if (!toplevel && !_ctx.refinement.mtkahypar.coarse_config_filename.empty()) {
      mt_kahypar_ctx = mt_kahypar_context_from_file(
          _ctx.refinement.mtkahypar.coarse_config_filename.c_str(), &error
      );
    } else {
      mt_kahypar_ctx = mt_kahypar_context_from_preset(DEFAULT);
    }
  } else {
    mt_kahypar_ctx =
        mt_kahypar_context_from_file(_ctx.refinement.mtkahypar.config_filename.c_str(), &error);
  }
  KASSERT(error.status == SUCCESS);

  mt_kahypar_set_partitioning_parameters(
      mt_kahypar_ctx, static_cast<mt_kahypar_partition_id_t>(p_ctx.k), p_ctx.epsilon(), CUT
  );
  mt_kahypar_set_seed(Random::get_seed());
  mt_kahypar_set_context_parameter(mt_kahypar_ctx, VERBOSE, "1", &error);
  KASSERT(error.status == SUCCESS);

  mt_kahypar_initialize(_ctx.parallel.num_threads, true);

  StaticArray<mt_kahypar_hypernode_weight_t> block_weights(p_ctx.k, static_array::noinit);
  p_graph.pfor_blocks([&](const BlockID b) { block_weights[b] = p_ctx.max_block_weight(b); });
  mt_kahypar_set_individual_target_block_weights(
      mt_kahypar_ctx, static_cast<mt_kahypar_partition_id_t>(p_ctx.k), block_weights.data()
  );

  const mt_kahypar_hypernode_id_t num_vertices = p_graph.n();
  const mt_kahypar_hyperedge_id_t num_edges = p_graph.m() / 2; // Only need one direction

  StaticArray<EdgeID> node_offsets(num_vertices + 1, static_array::noinit);
  reified(p_graph, [&](const auto &graph) {
    graph.pfor_nodes([&](const NodeID u) { node_offsets[u + 1] = graph.degree(u); });
  });
  node_offsets[0] = 0;
  parallel::prefix_sum(node_offsets.begin(), node_offsets.end(), node_offsets.begin());

  StaticArray<EdgeID> edge_position(2 * num_edges, static_array::noinit);
  reified(p_graph, [&](const auto &graph) {
    graph.pfor_nodes([&](const NodeID u) {
      EdgeID e = node_offsets[u];
      graph.adjacent_nodes(u, [&](const NodeID v) { edge_position[e++] = u < v; });
    });
  });
  parallel::prefix_sum(edge_position.begin(), edge_position.end(), edge_position.begin());

  StaticArray<mt_kahypar_hypernode_id_t> edges(2 * num_edges, static_array::noinit);
  StaticArray<mt_kahypar_hypernode_weight_t> edge_weights(num_edges, static_array::noinit);
  StaticArray<mt_kahypar_hypernode_weight_t> vertex_weights(num_vertices, static_array::noinit);

  reified(p_graph, [&](const auto &graph) {
    graph.pfor_nodes([&](const NodeID u) {
      vertex_weights[u] = static_cast<mt_kahypar_hypernode_weight_t>(graph.node_weight(u));

      EdgeID e = node_offsets[u];
      graph.adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        // Only need edges in one direction
        if (u < v) {
          const EdgeID position = edge_position[e] - 1;
          edges[2 * position] = static_cast<mt_kahypar_hypernode_id_t>(u);
          edges[2 * position + 1] = static_cast<mt_kahypar_hypernode_id_t>(v);
          edge_weights[position] = static_cast<mt_kahypar_hypernode_weight_t>(w);
        }

        e += 1;
      });
    });
  });

  // Node offsets and edge positions are not needed anymore; thus, free them to save memory.
  node_offsets.free();
  edge_position.free();

  mt_kahypar_hypergraph_t mt_kahypar_graph = mt_kahypar_create_graph(
      mt_kahypar_ctx,
      num_vertices,
      num_edges,
      edges.data(),
      edge_weights.data(),
      vertex_weights.data(),
      &error
  );
  KASSERT(error.status == SUCCESS);

  DBG << "Partition metrics before Mt-KaHyPar refinement: cut=" << metrics::edge_cut(p_graph)
      << " imbalance=" << metrics::imbalance(p_graph);

  StaticArray<mt_kahypar_partition_id_t> partition(num_vertices, static_array::noinit);
  reified(p_graph, [&](const auto &graph) {
    graph.pfor_nodes([&](const NodeID u) { partition[u] = p_graph.block(u); });
  });

  mt_kahypar_partitioned_hypergraph_t mt_kahypar_partitioned_graph =
      mt_kahypar_create_partitioned_hypergraph(
          mt_kahypar_graph,
          mt_kahypar_ctx,
          static_cast<mt_kahypar_partition_id_t>(p_ctx.k),
          partition.data(),
          &error
      );
  KASSERT(error.status == SUCCESS);

  // Run refinement
  mt_kahypar_improve_partition(mt_kahypar_partitioned_graph, mt_kahypar_ctx, 1, &error);
  KASSERT(error.status == SUCCESS);

  // Copy partition back to our graph
  StaticArray<mt_kahypar_partition_id_t> improved_partition(num_vertices, static_array::noinit);
  mt_kahypar_get_partition(mt_kahypar_partitioned_graph, improved_partition.data());
  reified(p_graph, [&](const auto &graph) {
    graph.pfor_nodes([&](const NodeID u) { p_graph.set_block(u, improved_partition[u]); });
  });

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
