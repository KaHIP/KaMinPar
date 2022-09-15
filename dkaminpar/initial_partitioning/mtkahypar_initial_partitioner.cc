/*******************************************************************************
 * @file:   mtkahypar_initial_partitioner.cc
 * @author: Daniel Seemaier
 * @date:   15.09.2022
 * @brief:  Initial partitioner that uses Mt-KaHypar. Only available if the
 * Mt-KaHyPar library is installed on the system.
 ******************************************************************************/
#include "dkaminpar/initial_partitioning/mtkahypar_initial_partitioner.h"

#include <kassert/kassert.hpp>

#include "kaminpar/partitioning_scheme/partitioning.h"

#include "common/assert.h"
#include "common/logger.h"
#include "common/noinit_vector.h"
#include "common/parallel/algorithm.h"
#include "common/timer.h"

namespace kaminpar::dist {
shm::PartitionedGraph MtKaHyParInitialPartitioner::initial_partition(const shm::Graph& graph) {
#ifdef KAMINPAR_HAS_MTKAHYPAR_LIB
    mt_kahypar_initialize_thread_pool(_ctx.parallel.num_threads, true);

    mt_kahypar_context_t* mtkahypar_ctx = mt_kahypar_context_new();
    mt_kahypar_configure_context_from_file(mtkahypar_context, _ctx.initial_partitioning.mtkahypar.preset_filename);

    // Setup graph for Mt-KaHyPar
    const mt_kahypar_hypernode_id_t num_vertices = graph.n();
    const mt_kahypar_hyperedge_id_t num_edges    = graph.m() / 2;          // Only need one direction
    const double                    imbalance    = _ctx.partition.epsilon; // @todo adjust epsilon
    const mt_kahypar_partition_id_t k            = _ctx.partition.k;

    // Copy node weights
    NoinitVector<mt_kahypar_hypernode_weight_t> node_weights(num_vertices);
    graph.pfor_nodes([&](const NodeID u) { node_weights[u] = graph.node_weight(u); });

    // Abuse edge_indices initially to build a prefix sum over the new node degrees
    NoinitVector<std::size_t> edge_indices(std::max<std::size_t>(num_vertices, num_edges) + 1);
    graph.pfor_nodes([&](const NodeID u) {
        const auto   adjacent_nodes = graph.adjacent_nodes(u);
        const EdgeID degree =
            std::count_if(adjacent_nodes.begin(), adjacent_nodes.end(), [u](const NodeID v) { return u < v; });
        edge_indices[u + 1] = degree;
    });
    parallel::prefix_sum(edge_indices.begin(), edge_indices.end(), edge_indices.begin());

    // Copy edge weights and egdes
    NoinitVector<mt_kahypar_hyperedge_weight_t[]> edge_weights(num_edges);
    NoinitVector<mt_kahypar_hyperedge_id_t[]>     edges(2 * num_edges);
    graph.pfor_nodes([&](const NodeID u) {
        for (const auto& [e, v]: graph.neighbors(u)) {
            if (u < v) {
                edge_weights[edge_indices[u]]  = graph.edge_weight(e);
                edges[2 * edge_indices[u]]     = u;
                edges[2 * edge_indices[u] + 1] = v;
                ++edge_indices[u];
            }
        }
    });

    // Build actual edge indices
    tbb::parallel_for<std::size_t>(0, num_edges + 1, [&](const std::size_t i) { edge_indices[i] = 2 * i; });

    NoinitVector<mt_kahypar_block_id_t> partition(num_vertices);
    mt_kahypar_hyperedge_weight_t objective = 0;

    mt_kahypar_partition(
        num_vertices, num_edges, imbalance, k, _ctx.seed, node_weights.data(), edge_weights.data(), edge_indices.data(),
        edges.data(), &objective, mtkahypar_context, partition.data(), false
    );

    //auto p_graph = shm::partitioning::partition(graph, shm_ctx);

    //return p_graph;
#else  // KAMINPAR_HAS_MTKAHYPAR_LIB
    KASSERT(false, "Mt-KaHyPar initial partitioner is not available.", assert::always);
    __builtin_unreachable();
#endif // KAMINPAR_HAS_MTKAHYPAR_LIB
}
} // namespace kaminpar::dist
