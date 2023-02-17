/*******************************************************************************
 * @file:   mtkahypar_initial_partitioner.cc
 * @author: Daniel Seemaier
 * @date:   15.09.2022
 * @brief:  Initial partitioner that uses Mt-KaHypar. Only available if the
 * Mt-KaHyPar library is installed on the system.
 ******************************************************************************/
#include "dkaminpar/initial_partitioning/mtkahypar_initial_partitioner.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

#include <kassert/kassert.hpp>

#ifdef KAMINPAR_HAS_MTKAHYPAR_LIB
    #include <libmtkahypar.h>
#endif // KAMINPAR_HAS_MTKAHYPAR_LIB

#include "kaminpar/partitioning/partitioning.h"

#include "common/assertion_levels.h"
#include "common/logger.h"
#include "common/noinit_vector.h"
#include "common/parallel/algorithm.h"
#include "common/random.h"
#include "common/timer.h"

namespace kaminpar::dist {
shm::PartitionedGraph MtKaHyParInitialPartitioner::initial_partition(
    [[maybe_unused]] const shm::Graph& graph, [[maybe_unused]] const PartitionContext& p_ctx
) {
#ifdef KAMINPAR_HAS_MTKAHYPAR_LIB
    mt_kahypar_context_t* mt_kahypar_ctx = mt_kahypar_context_new();
    mt_kahypar_load_preset(mt_kahypar_ctx, HIGH_QUALITY);
    mt_kahypar_set_partitioning_parameters(
        mt_kahypar_ctx, static_cast<mt_kahypar_partition_id_t>(p_ctx.k), p_ctx.epsilon, KM1, Random::seed
    );

    mt_kahypar_initialize_thread_pool(_ctx.parallel.num_threads, true);

    const mt_kahypar_hypernode_id_t num_vertices = graph.n();
    const mt_kahypar_hyperedge_id_t num_edges    = graph.m() / 2; // Only need one direction

    NoinitVector<mt_kahypar_hypernode_id_t>     edges;
    NoinitVector<mt_kahypar_hypernode_weight_t> edge_weights;
    NoinitVector<mt_kahypar_hypernode_weight_t> vertex_weights;
    edges.reserve(2 * num_edges);
    edge_weights.reserve(num_edges);
    vertex_weights.reserve(num_vertices);

    graph.pfor_nodes([&](const NodeID u) {
        vertex_weights.push_back(graph.node_weight(u));
        for (const auto [e, v]: graph.neighbors(u)) {
            if (v < u) { // Only need edges in one direction
                continue;
            }

            edges.push_back(u);
            edges.push_back(v);
            edge_weights.push_back(graph.edge_weight(e));
        }
    });

    mt_kahypar_graph_t* mt_kahypar_graph =
        mt_kahypar_create_graph(num_vertices, num_edges, edges.data(), edge_weights.data(), vertex_weights.data());

    mt_kahypar_partitioned_graph_t* mt_kahypar_partitioned_graph =
        mt_kahypar_partition_graph(mt_kahypar_graph, mt_kahypar_ctx);

    NoinitVector<mt_kahypar_partition_id_t> partition(num_vertices);
    mt_kahypar_get_graph_partition(mt_kahypar_partitioned_graph, partition.data());

    // Copy partition to BlockID vector
    StaticArray<BlockID> partition_cpy(num_vertices);
    tbb::parallel_for<std::size_t>(0, num_vertices, [&](const std::size_t i) {
        partition_cpy[i] = static_cast<BlockID>(partition[i]);
    });

    mt_kahypar_free_partitioned_graph(mt_kahypar_partitioned_graph);
    mt_kahypar_free_graph(mt_kahypar_graph);
    mt_kahypar_free_context(mt_kahypar_ctx);

    return shm::PartitionedGraph(graph, p_ctx.k, std::move(partition_cpy), scalable_vector<BlockID>(p_ctx.k, 1));
#else  // KAMINPAR_HAS_MTKAHYPAR_LIB
    ((void)_ctx);
    KASSERT(false, "Mt-KaHyPar initial partitioner is not available.", assert::always);
    __builtin_unreachable();
#endif // KAMINPAR_HAS_MTKAHYPAR_LIB
}
} // namespace kaminpar::dist
