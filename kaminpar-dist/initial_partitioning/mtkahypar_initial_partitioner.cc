/*******************************************************************************
 * Initial partitioner that uses Mt-KaHypar. Only available if the Mt-KaHyPar
 * library is installed on the system.
 *
 * @file:   mtkahypar_initial_partitioner.cc
 * @author: Daniel Seemaier
 * @date:   15.09.2022
 ******************************************************************************/
#include "kaminpar-dist/initial_partitioning/mtkahypar_initial_partitioner.h"

#ifdef KAMINPAR_HAVE_MTKAHYPAR_LIB
#include <mtkahypar.h>

#include "kaminpar-common/datastructures/noinit_vector.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/random.h"
#endif // KAMINPAR_HAVE_MTKAHYPAR_LIB

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"

namespace kaminpar::dist {

shm::PartitionedGraph MtKaHyParInitialPartitioner::initial_partition(
    [[maybe_unused]] const shm::Graph &graph, [[maybe_unused]] const shm::PartitionContext &p_ctx
) {
#ifdef KAMINPAR_HAVE_MTKAHYPAR_LIB
  mt_kahypar_error_t error{};

  mt_kahypar_context_t *mt_kahypar_ctx = mt_kahypar_context_from_preset(DEFAULT);
  mt_kahypar_set_partitioning_parameters(
      mt_kahypar_ctx, static_cast<mt_kahypar_partition_id_t>(p_ctx.k), p_ctx.epsilon(), CUT
  );
  mt_kahypar_set_seed(Random::get_seed());
  mt_kahypar_set_context_parameter(mt_kahypar_ctx, VERBOSE, "0", &error);
  KASSERT(error.status == SUCCESS);

  mt_kahypar_initialize(_ctx.parallel.num_threads, true);

  const mt_kahypar_hypernode_id_t num_vertices = graph.n();
  const mt_kahypar_hyperedge_id_t num_edges = graph.m() / 2; // Only need one direction

  NoinitVector<EdgeID> node_offsets(num_vertices + 1);
  reified(graph, [&](const auto &graph) {
    graph.pfor_nodes([&](const NodeID u) { node_offsets[u + 1] = graph.degree(u); });
  });
  node_offsets[0] = 0;
  parallel::prefix_sum(node_offsets.begin(), node_offsets.end(), node_offsets.begin());

  NoinitVector<EdgeID> edge_position(2 * num_edges);
  reified(graph, [&](const auto &graph) {
    graph.pfor_nodes([&](const NodeID u) {
      EdgeID e = node_offsets[u];
      graph.adjacent_nodes(u, [&](const NodeID v) { edge_position[e++] = u < v; });
    });
  });
  parallel::prefix_sum(edge_position.begin(), edge_position.end(), edge_position.begin());

  NoinitVector<mt_kahypar_hypernode_id_t> edges(2 * num_edges);
  NoinitVector<mt_kahypar_hypernode_weight_t> edge_weights(num_edges);
  NoinitVector<mt_kahypar_hypernode_weight_t> vertex_weights(num_vertices);
  edges.reserve(2 * num_edges);
  edge_weights.reserve(num_edges);

  reified(graph, [&](const auto &graph) {
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
  node_offsets.clear();
  node_offsets.shrink_to_fit();

  edge_position.clear();
  edge_position.shrink_to_fit();

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

  mt_kahypar_partitioned_hypergraph_t mt_kahypar_partitioned_graph =
      mt_kahypar_partition(mt_kahypar_graph, mt_kahypar_ctx, &error);
  KASSERT(error.status == SUCCESS);

  NoinitVector<mt_kahypar_partition_id_t> partition(num_vertices);
  mt_kahypar_get_partition(mt_kahypar_partitioned_graph, partition.data());

  // Copy partition to BlockID vector
  StaticArray<BlockID> partition_cpy(num_vertices);
  tbb::parallel_for<std::size_t>(0, num_vertices, [&](const std::size_t i) {
    partition_cpy[i] = static_cast<BlockID>(partition[i]);
  });

  mt_kahypar_free_partitioned_hypergraph(mt_kahypar_partitioned_graph);
  mt_kahypar_free_hypergraph(mt_kahypar_graph);
  mt_kahypar_free_context(mt_kahypar_ctx);

  return {graph, p_ctx.k, std::move(partition_cpy)};
#else  // KAMINPAR_HAVE_MTKAHYPAR_LIB
  KASSERT(false, "Mt-KaHyPar initial partitioner is not available.", assert::always);
  __builtin_unreachable();
#endif // KAMINPAR_HAVE_MTKAHYPAR_LIB
}

} // namespace kaminpar::dist
