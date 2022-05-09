/*******************************************************************************
 * @file:   dkaminpar_graphgen.h
 *
 * @author: Daniel Seemaier
 * @date:   26.11.21
 * @brief:  In-memory graph generator using KaGen.
 ******************************************************************************/
#include "apps/dkaminpar_graphgen.h"

#include <kagen_library.h>
#include <tbb/parallel_sort.h>

#include "dkaminpar/coarsening/contraction_helper.h"
#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dkaminpar/mpi_wrapper.h"
#include "kaminpar/parallel/atomic.h"
#include "kaminpar/utils/random.h"
#include "kaminpar/utils/timer.h"

namespace dkaminpar::graphgen {
using namespace std::string_literals;

DEFINE_ENUM_STRING_CONVERSION(GeneratorType, generator_type) = {
    {GeneratorType::NONE, "none"},     {GeneratorType::GNM, "gnm"},     {GeneratorType::GNP, "gnp"},
    {GeneratorType::RGG2D, "rgg2d"},   {GeneratorType::RGG3D, "rgg3d"}, {GeneratorType::RDG2D, "rdg2d"},
    {GeneratorType::RDG3D, "rdg3d"},   {GeneratorType::RHG, "rhg"},     {GeneratorType::GRID2D, "grid2d"},
    {GeneratorType::GRID3D, "grid3d"},
};

using namespace kagen;

namespace {
SET_DEBUG(false);

DistributedGraph build_graph(const EdgeList& edge_list, scalable_vector<GlobalNodeID> node_distribution) {
    SCOPED_TIMER("Build graph from edge list");

    const auto [size, rank] = mpi::get_comm_info();
    const GlobalNodeID from = node_distribution[rank];
    const GlobalNodeID to   = node_distribution[rank + 1];
    KASSERT(from <= to, "", assert::always);

    const auto n = static_cast<NodeID>(to - from);

    // bucket sort nodes
    START_TIMER("Bucket sort");
    scalable_vector<Atomic<NodeID>> buckets(n);
    tbb::parallel_for<EdgeID>(0, edge_list.size(), [&](const EdgeID e) {
        const GlobalNodeID u = std::get<0>(edge_list[e]);
        KASSERT(from <= u, "", assert::always);
        KASSERT(u < to, "", assert::always);

        buckets[u - from].fetch_add(1, std::memory_order_relaxed);
    });
    shm::parallel::prefix_sum(buckets.begin(), buckets.end(), buckets.begin());
    STOP_TIMER();

    const EdgeID m = buckets.back();

    // build edges array
    START_TIMER("Build edges array");
    scalable_vector<EdgeID> edges(m);
    graph::GhostNodeMapper  ghost_node_mapper(node_distribution);
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
    });
    STOP_TIMER();

    auto mapped_ghost_nodes = TIMED_SCOPE("Finalize ghost node mapping") {
        return ghost_node_mapper.finalize();
    };

    // build nodes array
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
        std::move(mapped_ghost_nodes.ghost_owner),
        std::move(mapped_ghost_nodes.ghost_to_global),
        std::move(mapped_ghost_nodes.global_to_ghost),
        false,
        MPI_COMM_WORLD};
    KASSERT(graph::debug::validate(graph), "", assert::heavy);
    return graph;
}

scalable_vector<GlobalNodeID> build_node_distribution(const std::pair<SInt, SInt> range) {
    const auto [size, rank] = mpi::get_comm_info();
    const GlobalNodeID to   = range.second;

    scalable_vector<GlobalNodeID> node_distribution(size + 1);
    mpi::allgather(&to, 1, node_distribution.data() + 1, 1);
    return node_distribution;
}

KaGen create_generator_object(const GeneratorContext ctx) {
    KaGen gen(MPI_COMM_WORLD);
    gen.SetSeed(ctx.seed);
    if (ctx.validate_graph) {
        gen.EnableUndirectedGraphVerification();
    }
    gen.EnableAdvancedStatistics();
    return gen;
}

KaGenResult create_rgg2d(const GeneratorContext ctx) {
    const GlobalEdgeID m      = (static_cast<GlobalEdgeID>(1) << ctx.m) * ctx.scale;
    const double       radius = ctx.r / std::sqrt(ctx.scale);
    const GlobalNodeID n      = static_cast<GlobalNodeID>(std::sqrt(1.0 * m / M_PI) / radius);
    return create_generator_object(ctx).GenerateRGG2D(n, radius);
}

KaGenResult create_rgg3d(const GeneratorContext ctx) {
    const GlobalEdgeID m      = (static_cast<GlobalEdgeID>(1) << ctx.m) * ctx.scale;
    const double       radius = ctx.r / std::cbrt(ctx.scale);
    const GlobalNodeID n      = static_cast<GlobalNodeID>(std::sqrt(3.0 / 4.0 * m / M_PI / (radius * radius * radius)));
    return create_generator_object(ctx).GenerateRGG3D(n, radius);
}

KaGenResult create_rhg(const GeneratorContext ctx) {
    const GlobalNodeID m = (static_cast<GlobalNodeID>(1) << ctx.m) * ctx.scale;
    const GlobalNodeID n = m / ctx.d;
    return create_generator_object(ctx).GenerateRHG(n, ctx.gamma, ctx.d);
}

KaGenResult create_grid2d(const GeneratorContext ctx) {
    const GlobalNodeID n = (static_cast<GlobalNodeID>(1) << ctx.n) * ctx.scale;
    return create_generator_object(ctx).GenerateGrid2D(n, ctx.p);
}

KaGenResult create_grid3d(const GeneratorContext ctx) {
    const GlobalNodeID n = (static_cast<GlobalNodeID>(1) << ctx.n) * ctx.scale;
    return create_generator_object(ctx).GenerateGrid3D(n, ctx.p);
}
} // namespace

DistributedGraph generate(const GeneratorContext ctx) {
    const auto [edges, range] = [&] {
        switch (ctx.type) {
            case GeneratorType::NONE:
                FATAL_ERROR << "no graph generator configured";
                break;

            case GeneratorType::RGG2D:
                return create_rgg2d(ctx);

            case GeneratorType::RGG3D:
                return create_rgg3d(ctx);

            case GeneratorType::RHG:
                return create_rhg(ctx);

            case GeneratorType::GRID2D:
                return create_grid2d(ctx);

            case GeneratorType::GRID3D:
                return create_grid3d(ctx);

            default:
                FATAL_ERROR << "graph generator is deactivated";
        }
        __builtin_unreachable();
    }();

    return build_graph(edges, build_node_distribution(range));
}
} // namespace dkaminpar::graphgen
