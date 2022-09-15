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

#include "kaminpar/partitioning_scheme/partitioning.h"

#include "common/assert.h"
#include "common/logger.h"
#include "common/noinit_vector.h"
#include "common/parallel/algorithm.h"
#include "common/timer.h"

namespace kaminpar::dist {
namespace {
const char* default_flows_preset = R"(
# general
maxnet-removal-factor=0.01
smallest-maxnet-threshold=50000
maxnet-ignore=1000
num-vcycles=0
# main -> shared_memory
s-use-localized-random-shuffle=false
s-static-balancing-work-packages=128
# main -> preprocessing
p-enable-community-detection=true
# main -> preprocessing -> community_detection
p-louvain-edge-weight-function=hybrid
p-max-louvain-pass-iterations=5
p-louvain-min-vertex-move-fraction=0.01
p-vertex-degree-sampling-threshold=200000
# main -> coarsening
c-type=multilevel_coarsener
c-use-adaptive-edge-size=true
c-min-shrink-factor=1.01
c-max-shrink-factor=2.5
c-s=1
c-t=160
c-vertex-degree-sampling-threshold=200000
# main -> coarsening -> rating
c-rating-score=heavy_edge
c-rating-heavy-node-penalty=no_penalty
c-rating-acceptance-criterion=best_prefer_unmatched
# main -> initial_partitioning
i-mode=rb
i-runs=20
i-use-adaptive-ip-runs=true
i-min-adaptive-ip-runs=5
i-perform-refinement-on-best-partitions=true
i-fm-refinement-rounds=1
i-lp-maximum-iterations=20
i-lp-initial-block-size=5
# main -> initial_partitioning -> refinement
i-r-refine-until-no-improvement=false
i-r-relative-improvement-threshold=0.0
# main -> initial_partitioning -> refinement -> label_propagation
i-r-lp-type=label_propagation_km1
i-r-lp-maximum-iterations=5
i-r-lp-rebalancing=true
i-r-lp-he-size-activation-threshold=100
# main -> initial_partitioning -> refinement -> fm
i-r-fm-type=fm_gain_cache
i-r-fm-multitry-rounds=5
i-r-fm-perform-moves-global=false
i-r-fm-rollback-parallel=true
i-r-fm-rollback-balance-violation-factor=1
i-r-fm-seed-nodes=25
i-r-fm-obey-minimal-parallelism=false
i-r-fm-release-nodes=true
i-r-fm-time-limit-factor=0.25
i-r-fm-iter-moves-on-recalc=true
# main -> initial_partitioning -> refinement -> flows
i-r-flow-algo=do_nothing
# main -> refinement
r-refine-until-no-improvement=true
r-relative-improvement-threshold=0.0025
# main -> refinement -> label_propagation
r-lp-type=label_propagation_km1
r-lp-maximum-iterations=5
r-lp-rebalancing=true
r-lp-he-size-activation-threshold=100
# main -> refinement -> fm
r-fm-type=fm_gain_cache
r-fm-multitry-rounds=10
r-fm-perform-moves-global=false
r-fm-rollback-parallel=true
r-fm-rollback-balance-violation-factor=1.25
r-fm-seed-nodes=25
r-fm-release-nodes=true
r-fm-min-improvement=-1.0
r-fm-obey-minimal-parallelism=true
r-fm-time-limit-factor=0.25
r-fm-iter-moves-on-recalc=true
# main -> refinement -> flows
r-flow-algo=flow_cutter
r-flow-scaling=16
r-flow-max-num-pins=4294967295
r-flow-find-most-balanced-cut=true
r-flow-determine-distance-from-cut=true
r-flow-parallel-search-multiplier=1.0
r-flow-max-bfs-distance=2
r-flow-min-relative-improvement-per-round=0.001
r-flow-time-limit-factor=8
r-flow-skip-small-cuts=true
r-flow-skip-unpromising-blocks=true
r-flow-pierce-in-bulk=true
)";
}

shm::PartitionedGraph MtKaHyParInitialPartitioner::initial_partition(const shm::Graph& graph) {
#ifdef KAMINPAR_HAS_MTKAHYPAR_LIB
    mt_kahypar_initialize_thread_pool(_ctx.parallel.num_threads, true);

    mt_kahypar_context_t* mtkahypar_ctx = mt_kahypar_context_new();
    if (_ctx.initial_partitioning.mtkahypar.preset_filename.empty()) {
        char filename[] = "kaminpar_mtkahypar_ip.XXXXXX";
        {
            int fd = mkstemp(filename);
            if (fd < 0) {
                FATAL_ERROR << "Could not create a temporary file for the Mt-KaHyPar preset";
            }
            FILE* file = fdopen(fd, "r");
            fputs(default_flows_preset, file);
            fclose(file);
        }
        mt_kahypar_configure_context_from_file(mtkahypar_ctx, filename);
    } else {
        mt_kahypar_configure_context_from_file(
            mtkahypar_ctx, _ctx.initial_partitioning.mtkahypar.preset_filename.c_str()
        );
    }

    // Setup graph for Mt-KaHyPar
    const mt_kahypar_hypernode_id_t num_vertices = graph.n();
    const mt_kahypar_hyperedge_id_t num_edges    = graph.m() / 2; // Only need one direction
    const mt_kahypar_partition_id_t k            = _ctx.partition.k;

    double imbalance = 0;
    for (BlockID b = 0; b < k; ++b) {
        imbalance = std::max<double>(
            imbalance,
            1.0 * _ctx.partition.max_block_weight(b) / _ctx.partition.perfectly_balanced_block_weight(b) - 1.0
        );
    }

    // Copy node weights
    NoinitVector<mt_kahypar_hypernode_weight_t> node_weights(num_vertices);
    graph.pfor_nodes([&](const NodeID u) { node_weights[u] = graph.node_weight(u); });

    // Abuse edge_indices initially to build a prefix sum over the new node degrees
    NoinitVector<std::size_t> edge_indices(std::max<std::size_t>(num_vertices, num_edges) + 1);
    edge_indices.front() = 0;
    graph.pfor_nodes([&](const NodeID u) {
        const auto   adjacent_nodes = graph.adjacent_nodes(u);
        const EdgeID degree =
            std::count_if(adjacent_nodes.begin(), adjacent_nodes.end(), [u](const NodeID v) { return u < v; });
        edge_indices[u + 1] = degree;
    });
    parallel::prefix_sum(edge_indices.begin(), edge_indices.end(), edge_indices.begin());

    // Copy edge weights and egdes
    NoinitVector<mt_kahypar_hyperedge_weight_t> edge_weights(num_edges);
    NoinitVector<mt_kahypar_hyperedge_id_t>     edges(2 * num_edges);
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
    edge_indices.resize(num_edges + 1);
    tbb::parallel_for<std::size_t>(0, num_edges + 1, [&](const std::size_t i) { edge_indices[i] = 2 * i; });

    NoinitVector<mt_kahypar_partition_id_t> partition(num_vertices);
    tbb::parallel_for<std::size_t>(0, partition.size(), [&partition](const std::size_t i) { partition[i] = -1; });
    mt_kahypar_hyperedge_weight_t objective = 0;

    mt_kahypar_partition(
        num_vertices, num_edges, imbalance, k, _ctx.seed, node_weights.data(), edge_weights.data(), edge_indices.data(),
        edges.data(), &objective, mtkahypar_ctx, partition.data(), false
    );

    // Copy partition to BlockID vector
    StaticArray<BlockID> partition_cpy(num_vertices);
    tbb::parallel_for<std::size_t>(0, num_vertices, [&](const std::size_t i) {
        partition_cpy[i] = static_cast<BlockID>(partition[i]);
    });

    scalable_vector<BlockID> final_ks(k);
    tbb::parallel_for<std::size_t>(0, k, [&final_ks](const std::size_t i) { final_ks[i] = 1; });

    return shm::PartitionedGraph(graph, static_cast<BlockID>(k), std::move(partition_cpy), std::move(final_ks));
#else  // KAMINPAR_HAS_MTKAHYPAR_LIB
    KASSERT(false, "Mt-KaHyPar initial partitioner is not available.", assert::always);
    __builtin_unreachable();
#endif // KAMINPAR_HAS_MTKAHYPAR_LIB
}
} // namespace kaminpar::dist
