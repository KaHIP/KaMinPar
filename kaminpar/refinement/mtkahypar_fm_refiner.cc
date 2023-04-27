#include "kaminpar/refinement/mtkahypar_fm_refiner.h"

#include "kaminpar/context.h"
#include "kaminpar/datastructures/partitioned_graph.h"

#include "common/logger.h"

#ifdef KAMINPAR_HAVE_MTKAHYPAR_LIB
#include <libmtkahypar.h>
#endif // KAMINPAR_HAVE_MTKAHYPAR_LIB

namespace kaminpar::shm {
MtKaHyParFMRefiner::MtKaHyParFMRefiner(const Context &ctx) : _ctx(ctx) {}

bool MtKaHyParFMRefiner::refine(
    [[maybe_unused]] PartitionedGraph &p_graph,
    [[maybe_unused]] const PartitionContext &p_ctx
) {
#ifdef KAMINPAR_HAVE_MTKAHYPAR_LIB
  mt_kahypar_context_t *mt_kahypar_ctx = mt_kahypar_context_new();
  mt_kahypar_load_preset(mt_kahypar_ctx, SPEED);
  mt_kahypar_set_partitioning_parameters(
      mt_kahypar_ctx,
      static_cast<mt_kahypar_partition_id_t>(p_ctx.k),
      p_ctx.epsilon,
      KM1,
      Random::seed
  );
  mt_kahypar_set_context_parameter(mt_kahypar_ctx, VERBOSE, "0");

  mt_kahypar_initialize_thread_pool(_ctx.parallel.num_threads, true);

  const mt_kahypar_hypernode_id_t num_vertices = graph.n();
  const mt_kahypar_hyperedge_id_t num_edges =
      graph.m() / 2; // Only need one direction

  NoinitVector<EdgeID> edge_position(2 * num_edges);
  graph.pfor_nodes([&](const NodeID u) {
    for (const auto [e, v] : graph.neighbors(u)) {
      edge_position[e] = u < v;
    }
  });
  parallel::prefix_sum(
      edge_position.begin(), edge_position.end(), edge_position.begin()
  );

  NoinitVector<mt_kahypar_hypernode_id_t> edges(2 * num_edges);
  NoinitVector<mt_kahypar_hypernode_weight_t> edge_weights(num_edges);
  NoinitVector<mt_kahypar_hypernode_weight_t> vertex_weights(num_vertices);
  edges.reserve(2 * num_edges);
  edge_weights.reserve(num_edges);

  graph.pfor_nodes([&](const NodeID u) {
    vertex_weights[u] =
        static_cast<mt_kahypar_hypernode_weight_t>(graph.node_weight(u));

    for (const auto [e, v] : graph.neighbors(u)) {
      if (v < u) { // Only need edges in one direction
        continue;
      }

      EdgeID position = edge_position[e] - 1;
      edges[2 * position] = static_cast<mt_kahypar_hypernode_id_t>(u);
      edges[2 * position + 1] = static_cast<mt_kahypar_hypernode_id_t>(v);
      edge_weights[position] =
          static_cast<mt_kahypar_hypernode_weight_t>(graph.edge_weight(e));
    }
  });

  mt_kahypar_graph_t *mt_kahypar_graph = mt_kahypar_create_graph(
      num_vertices,
      num_edges,
      edges.data(),
      edge_weights.data(),
      vertex_weights.data()
  );

  mt_kahypar_partitioned_graph_t *mt_kahypar_partitioned_graph =
      mt_kahypar_partition_graph(mt_kahypar_graph, mt_kahypar_ctx);

  NoinitVector<mt_kahypar_partition_id_t> partition(num_vertices);
  mt_kahypar_get_graph_partition(
      mt_kahypar_partitioned_graph, partition.data()
  );

  // Copy partition to BlockID vector
  StaticArray<BlockID> partition_cpy(num_vertices);
  tbb::parallel_for<std::size_t>(0, num_vertices, [&](const std::size_t i) {
    partition_cpy[i] = static_cast<BlockID>(partition[i]);
  });

  mt_kahypar_free_partitioned_graph(mt_kahypar_partitioned_graph);
  mt_kahypar_free_graph(mt_kahypar_graph);
  mt_kahypar_free_context(mt_kahypar_ctx);

  return shm::PartitionedGraph(
      graph, p_ctx.k, std::move(partition_cpy), std::vector<BlockID>(p_ctx.k, 1)
  );
#else  // KAMINPAR_HAVE_MTKAHYPAR_LIB
  LOG_WARNING << "Mt-KaHyPar is not available; skipping refinement";
  return false;
#endif // KAMINPAR_HAVE_MTKAHYPAR_LIB
}
} // namespace kaminpar::shm
