/*******************************************************************************
 * @file:   graphgen.cc
 * @author: Daniel Seemaier
 * @date:   26.11.21
 * @brief:  In-memory graph generator using KaGen.
 ******************************************************************************/
#include "apps/dkaminpar/graphgen.h"

#ifdef KAMINPAR_GRAPHGEN
    #include <kagen.h>
#endif // KAMINPAR_GRAPHGEN

#include <tbb/parallel_sort.h>

#include "dkaminpar/coarsening/contraction_helper.h"
#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_graph_builder.h"
#include "dkaminpar/definitions.h"
#include "dkaminpar/growt.h"
#include "dkaminpar/mpi/wrapper.h"

#include "kaminpar/graphutils/graph_permutation.h"
#include "kaminpar/graphutils/graph_rearrangement.h"

#include "common/datastructures/marker.h"
#include "common/parallel/algorithm.h"
#include "common/parallel/atomic.h"
#include "common/random.h"
#include "common/timer.h"

namespace kaminpar::dist {
#ifdef KAMINPAR_GRAPHGEN
using namespace kagen;

namespace {
SET_DEBUG(false);

DistributedGraph build_graph(
    const EdgeList& edge_list, scalable_vector<GlobalNodeID> node_distribution,
    const VertexWeights& original_vertex_weights, const EdgeWeights& original_edge_weights
) {
    SCOPED_TIMER("Build graph from edge list");

    const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
    const GlobalNodeID from = node_distribution[rank];
    const GlobalNodeID to   = node_distribution[rank + 1];
    KASSERT(from <= to, "", assert::always);

    const auto n = static_cast<NodeID>(to - from);

    // Bucket sort nodes
    START_TIMER("Bucket sort");
    scalable_vector<parallel::Atomic<NodeID>> buckets(n);
    tbb::parallel_for<EdgeID>(0, edge_list.size(), [&](const EdgeID e) {
        const GlobalNodeID u = std::get<0>(edge_list[e]);
        KASSERT(from <= u, "", assert::always);
        KASSERT(u < to, "", assert::always);

        buckets[u - from].fetch_add(1, std::memory_order_relaxed);
    });
    parallel::prefix_sum(buckets.begin(), buckets.end(), buckets.begin());
    STOP_TIMER();

    const EdgeID m                = buckets.back();
    const bool   has_node_weights = !original_vertex_weights.empty();
    const bool   has_edge_weights = !original_edge_weights.empty();

    // Build vertex weights array
    START_TIMER("Build node weights array");
    scalable_vector<NodeWeight> node_weights(has_node_weights ? n : 0);
    std::transform(
        original_vertex_weights.begin(), original_vertex_weights.end(), node_weights.begin(),
        [](const auto vertex_weight) { return static_cast<NodeWeight>(vertex_weight); }
    );
    STOP_TIMER();

    // Build edges array
    START_TIMER("Build edges array");
    scalable_vector<EdgeID>     edges(m);
    scalable_vector<EdgeWeight> edge_weights(has_edge_weights ? m : 0);

    graph::GhostNodeMapper ghost_node_mapper(MPI_COMM_WORLD, node_distribution);
    tbb::parallel_for<EdgeID>(0, edge_list.size(), [&](const EdgeID e) {
        const auto [u, v] = edge_list[e];
        KASSERT(from <= u, "", assert::always);
        KASSERT(u < to, "", assert::always);

        const auto pos = buckets[u - from].fetch_sub(1, std::memory_order_relaxed) - 1;
        KASSERT(pos < edges.size());

        if (from <= v && v < to) {
            edges[pos] = static_cast<NodeID>(v - from);
        } else {
            edges[pos] = ghost_node_mapper.new_ghost_node(v);
        }

        if (has_edge_weights) {
            edge_weights[pos] = static_cast<EdgeWeight>(original_edge_weights[e]);
        }
    });
    STOP_TIMER();

    auto mapped_ghost_nodes = TIMED_SCOPE("Finalize ghost node mapping") {
        return ghost_node_mapper.finalize();
    };

    // Build nodes array
    START_TIMER("Build nodes array");
    scalable_vector<NodeID> nodes(n + 1);
    tbb::parallel_for<NodeID>(0, n, [&](const NodeID u) { nodes[u] = buckets[u]; });
    nodes.back() = m;
    STOP_TIMER();

    DistributedGraph graph{
        std::move(node_distribution),
        mpi::build_distribution_from_local_count<GlobalEdgeID, scalable_vector>(m, MPI_COMM_WORLD),
        std::move(nodes),
        std::move(edges),
        std::move(node_weights),
        std::move(edge_weights),
        std::move(mapped_ghost_nodes.ghost_owner),
        std::move(mapped_ghost_nodes.ghost_to_global),
        std::move(mapped_ghost_nodes.global_to_ghost),
        false,
        MPI_COMM_WORLD};
    KASSERT(graph::debug::validate(graph), "", assert::heavy);
    return graph;
}

scalable_vector<GlobalNodeID> build_node_distribution(const std::pair<SInt, SInt> range) {
    const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
    const GlobalNodeID to   = range.second;

    scalable_vector<GlobalNodeID> node_distribution(size + 1);
    mpi::allgather(&to, 1, node_distribution.data() + 1, 1, MPI_COMM_WORLD);
    return node_distribution;
}
} // namespace
#endif // KAMINPAR_GRAPHGEN

DistributedGraph generate([[maybe_unused]] const std::string& properties) {
#ifdef KAMINPAR_GRAPHGEN
    auto result = [&] {
        KaGen kagen(MPI_COMM_WORLD);
        kagen.EnableOutput(false);
        kagen.EnableBasicStatistics();
        return kagen.GenerateFromOptionString(properties);
    }();

    return build_graph(
        std::move(result.edges), build_node_distribution(result.vertex_range), std::move(result.vertex_weights),
        std::move(result.edge_weights)
    );
#endif // KAMINPAR_GRAPHGEN

    throw std::runtime_error("graph generators are unavailable");
}

std::string generate_filename(const std::string& properties) {
    return std::string("kagen_") + properties;
}
} // namespace kaminpar::dist
