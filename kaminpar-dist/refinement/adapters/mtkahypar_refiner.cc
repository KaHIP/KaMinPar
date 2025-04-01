/*******************************************************************************
 * Adapter to use Mt-KaHyPar as a refinement algorithm.
 *
 * @file:   mtkahypar_refiner.cc
 * @author: Daniel Seemaier
 * @date:   17.10.2023
 ******************************************************************************/
#include "kaminpar-dist/refinement/adapters/mtkahypar_refiner.h"

#ifdef KAMINPAR_HAVE_MTKAHYPAR_LIB
#include <mtkahypar.h>

#include "kaminpar-mpi/wrapper.h"

#include "kaminpar-dist/graphutils/replicator.h"

#include "kaminpar-shm/metrics.h"

#include "kaminpar-common/asserting_cast.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/random.h"
#endif // KAMINPAR_HAVE_MTKAHYPAR_LIB

#include "kaminpar-dist/logger.h"

namespace kaminpar::dist {

namespace {

SET_STATISTICS_FROM_GLOBAL();
SET_DEBUG(true);

} // namespace

MtKaHyParRefinerFactory::MtKaHyParRefinerFactory(const Context &ctx) : _ctx(ctx) {}

std::unique_ptr<GlobalRefiner> MtKaHyParRefinerFactory::create(
    DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
) {
  return std::make_unique<MtKaHyParRefiner>(_ctx, p_graph, p_ctx);
}

MtKaHyParRefiner::MtKaHyParRefiner(
    const Context &ctx, DistributedPartitionedGraph &p_graph, const PartitionContext &p_ctx
)
    : _ctx(ctx),
      _p_graph(p_graph),
      _p_ctx(p_ctx) {}

void MtKaHyParRefiner::initialize() {}

bool MtKaHyParRefiner::refine() {
#ifdef KAMINPAR_HAVE_MTKAHYPAR_LIB
  auto shm_graph_pair = allgather_graph(_p_graph);
  auto &shm_graph = shm_graph_pair.first;
  auto &shm_p_graph = shm_graph_pair.second;

  // If we only run MtKaHyPar on the root PE, free the gathered graph on all other PEs
  const bool participate = !_ctx.refinement.mtkahypar.only_run_on_root ||
                           mpi::get_comm_rank(_p_graph.communicator()) == 0;
  if (!participate) {
    [[maybe_unused]] auto _ = std::move(shm_graph_pair);
  }

  mpi::barrier(_p_graph.communicator());

  if (participate) {
    mt_kahypar_error_t error{};

    mt_kahypar_context_t *mt_kahypar_ctx;
    if (_ctx.refinement.mtkahypar.config_filename.empty()) {
      const bool toplevel = (_p_graph.n() == _ctx.partition.n);

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
    mt_kahypar_set_partitioning_parameters(
        mt_kahypar_ctx, static_cast<mt_kahypar_partition_id_t>(_p_ctx.k), _p_ctx.epsilon(), CUT
    );
    mt_kahypar_set_seed(Random::get_seed());

    StaticArray<mt_kahypar_hypernode_weight_t> block_weights(_p_ctx.k, static_array::noinit);
    shm_p_graph->pfor_blocks([&](const BlockID b) {
      block_weights[b] = asserting_cast<mt_kahypar_hypernode_weight_t>(_p_ctx.max_block_weight(b));
    });
    mt_kahypar_set_individual_target_block_weights(
        mt_kahypar_ctx, asserting_cast<mt_kahypar_partition_id_t>(_p_ctx.k), block_weights.data()
    );

    mt_kahypar_set_context_parameter(mt_kahypar_ctx, VERBOSE, "1", &error);
    mt_kahypar_initialize(_ctx.parallel.num_threads, true);

    const auto num_vertices = asserting_cast<mt_kahypar_hypernode_id_t>(shm_graph->n());
    const auto num_edges = asserting_cast<mt_kahypar_hyperedge_id_t>(shm_graph->m() / 2);

    StaticArray<EdgeID> node_offsets(num_vertices + 1, static_array::noinit);
    reified(*shm_graph, [&](const auto &graph) {
      graph.pfor_nodes([&](const NodeID u) { node_offsets[u + 1] = graph.degree(u); });
    });
    node_offsets[0] = 0;
    parallel::prefix_sum(node_offsets.begin(), node_offsets.end(), node_offsets.begin());

    StaticArray<EdgeID> edge_position(2 * num_edges, static_array::noinit);
    reified(*shm_graph, [&](const auto &graph) {
      graph.pfor_nodes([&](const NodeID u) {
        EdgeID e = node_offsets[u];
        graph.adjacent_nodes(u, [&](const NodeID v) { edge_position[e++] = u < v; });
      });
    });
    parallel::prefix_sum(edge_position.begin(), edge_position.end(), edge_position.begin());

    StaticArray<mt_kahypar_hypernode_id_t> edges(2 * num_edges, static_array::noinit);
    StaticArray<mt_kahypar_hypernode_weight_t> edge_weights(num_edges, static_array::noinit);
    StaticArray<mt_kahypar_hypernode_weight_t> vertex_weights(num_vertices, static_array::noinit);

    reified(*shm_graph, [&](const auto &graph) {
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

    DBG << "Partition metrics before Mt-KaHyPar refinement: cut="
        << shm::metrics::edge_cut(*shm_p_graph)
        << " imbalance=" << shm::metrics::imbalance(*shm_p_graph);

    StaticArray<mt_kahypar_partition_id_t> mt_kahypar_original_partition(
        num_vertices, static_array::noinit
    );
    reified(*shm_graph, [&](const auto &graph) {
      graph.pfor_nodes([&](const NodeID u) {
        mt_kahypar_original_partition[u] =
            static_cast<mt_kahypar_partition_id_t>(shm_p_graph->block(u));
      });
    });

    mt_kahypar_partitioned_hypergraph_t mt_kahypar_partitioned_graph =
        mt_kahypar_create_partitioned_hypergraph(
            mt_kahypar_graph,
            mt_kahypar_ctx,
            static_cast<mt_kahypar_partition_id_t>(_p_ctx.k),
            mt_kahypar_original_partition.data(),
            &error
        );

    // Run refinement
    mt_kahypar_improve_partition(mt_kahypar_partitioned_graph, mt_kahypar_ctx, 1, &error);

    // Copy partition back to our graph
    StaticArray<mt_kahypar_partition_id_t> mt_kahypar_improved_partition(
        num_vertices, static_array::noinit
    );
    mt_kahypar_get_partition(mt_kahypar_partitioned_graph, mt_kahypar_improved_partition.data());
    reified(*shm_graph, [&](const auto &graph) {
      graph.pfor_nodes([&](const NodeID u) {
        shm_p_graph->set_block(u, asserting_cast<BlockID>(mt_kahypar_improved_partition[u]));
      });
    });

    DBG << "Partition metrics after Mt-KaHyPar refinement: cut="
        << shm::metrics::edge_cut(*shm_p_graph)
        << " imbalance=" << shm::metrics::imbalance(*shm_p_graph);

    // Free Mt-KaHyPar structs
    mt_kahypar_free_partitioned_hypergraph(mt_kahypar_partitioned_graph);
    mt_kahypar_free_hypergraph(mt_kahypar_graph);
    mt_kahypar_free_context(mt_kahypar_ctx);
  }

  // Copy global partition back to the distributed graph
  if (_ctx.refinement.mtkahypar.only_run_on_root) {
    if (participate) {
      _p_graph = distribute_partition(_p_graph.graph(), _p_ctx.k, shm_p_graph->raw_partition(), 0);
    } else {
      StaticArray<shm::BlockID> dummy;
      _p_graph = distribute_partition(_p_graph.graph(), _p_ctx.k, dummy, 0);
    }
  } else {
    _p_graph = distribute_best_partition(_p_graph.graph(), std::move(*shm_p_graph));
  }

  return false;
#else  // KAMINPAR_HAVE_MTKAHYPAR_LIB
  LOG_WARNING << "Mt-KaHyPar is not available; skipping refinement";
  return false;
#endif // KAMINPAR_HAVE_MTKAHYPAR_LIB
}

} // namespace kaminpar::dist
