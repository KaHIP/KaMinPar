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
#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/datastructure/distributed_graph_builder.h"
#include "dkaminpar/definitions.h"
#include "dkaminpar/growt.h"
#include "dkaminpar/mpi/wrapper.h"
#include "kaminpar/datastructure/marker.h"
#include "kaminpar/graphutils/graph_permutation.h"
#include "kaminpar/graphutils/graph_rearrangement.h"
#include "kaminpar/parallel/algorithm.h"
#include "kaminpar/parallel/atomic.h"
#include "kaminpar/utils/random.h"
#include "kaminpar/utils/timer.h"

namespace dkaminpar::graphgen {
using namespace std::string_literals;

DEFINE_ENUM_STRING_CONVERSION(GeneratorType, generator_type) = {
    {GeneratorType::NONE, "none"},   {GeneratorType::GNM, "gnm"},       {GeneratorType::RGG2D, "rgg2d"},
    {GeneratorType::RGG3D, "rgg3d"}, {GeneratorType::RDG2D, "rdg2d"},   {GeneratorType::RDG3D, "rdg3d"},
    {GeneratorType::RHG, "rhg"},     {GeneratorType::GRID2D, "grid2d"}, {GeneratorType::GRID3D, "grid3d"},
    {GeneratorType::RMAT, "rmat"},
};

using namespace kagen;

namespace {
SET_DEBUG(false);

PEID find_global_node_owner(const GlobalNodeID node, const scalable_vector<GlobalNodeID>& node_distribution) {
    KASSERT(node < node_distribution.back());
    auto it = std::upper_bound(node_distribution.begin() + 1, node_distribution.end(), node);
    KASSERT(it != node_distribution.end());
    return static_cast<PEID>(std::distance(node_distribution.begin(), it) - 1);
}

growt::StaticGhostNodeMapping remap_ghost_nodes(
    const EdgeList& edge_list, const scalable_vector<NodeID>& local_old_to_new,
    const scalable_vector<GlobalNodeID>& node_distribution) {
    const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
    const GlobalNodeID from = node_distribution[rank];
    const GlobalNodeID to   = node_distribution[rank + 1];

    std::vector<int> sendcounts(size);
    std::vector<int> recvcounts(size);

    // Count number of messages for each PE
    shm::Marker<> counted_pe(size);
    GlobalNodeID  current_global_u = kInvalidGlobalNodeID;

    for (const auto& [global_u, global_v]: edge_list) {
        if (from <= global_v && global_v < to) {
            continue; // Target is not a ghost node
        }

        // Start of new node
        if (global_u != current_global_u) {
            counted_pe.reset();
            current_global_u = global_u;
        }

        // Count owner PE
        const PEID owner = find_global_node_owner(global_v, node_distribution);
        if (!counted_pe.get(owner)) {
            ++sendcounts[owner];
            counted_pe.set(owner);
        }
    }

    // Exchange number of messages
    DBG << V(sendcounts);
    mpi::alltoall(sendcounts.data(), 1, recvcounts.data(), 1, MPI_COMM_WORLD);
    DBG << V(recvcounts);

    // Build send / receive displacements
    std::vector<int> sdispls(size + 1);
    std::partial_sum(sendcounts.begin(), sendcounts.end(), sdispls.begin() + 1);
    std::vector<int> rdispls(size + 1);
    std::partial_sum(recvcounts.begin(), recvcounts.end(), rdispls.begin() + 1);

    // Build messages to send
    struct RemapMessage {
        GlobalNodeID old_global_label;
        GlobalNodeID new_global_label;
    };

    const std::size_t         total_num_send_messages = std::accumulate(sendcounts.begin(), sendcounts.end(), 0u);
    std::vector<RemapMessage> sendbuf(total_num_send_messages);
    std::vector<int>          pos(size);

    current_global_u = kInvalidGlobalNodeID;
    for (const auto& [global_u, global_v]: edge_list) {
        if (from <= global_v && global_v < to) {
            continue; // Target is not a ghost node
        }

        // Start of new node
        if (global_u != current_global_u) {
            counted_pe.reset();
            current_global_u = global_u;
        }

        // Count owner PE
        const PEID owner = find_global_node_owner(global_v, node_distribution);
        if (!counted_pe.get(owner)) {
            const std::size_t index = sdispls[owner] + static_cast<std::size_t>(pos[owner]++);

            KASSERT(from <= global_u);
            KASSERT(global_u < to);
            const NodeID local_u = static_cast<NodeID>(global_u - from);

            sendbuf[index].old_global_label = global_u;
            sendbuf[index].new_global_label = from + local_old_to_new[local_u];

            counted_pe.set(owner);
        }
    }

    // Exchange messages
    const std::size_t         total_num_receive_messages = std::accumulate(recvcounts.begin(), recvcounts.end(), 0u);
    std::vector<RemapMessage> recvbuf(total_num_receive_messages);

    KASSERT(sendcounts.size() >= static_cast<std::size_t>(size));
    KASSERT(sdispls.size() >= static_cast<std::size_t>(size));
    KASSERT(recvcounts.size() >= static_cast<std::size_t>(size));
    KASSERT(rdispls.size() >= static_cast<std::size_t>(size));

    KASSERT(sendbuf.size() == static_cast<std::size_t>(sendcounts[size - 1] + sdispls[size - 1]));
    KASSERT(recvbuf.size() == static_cast<std::size_t>(recvcounts[size - 1] + rdispls[size - 1]));

    mpi::alltoallv(
        sendbuf.data(), sendcounts.data(), sdispls.data(), recvbuf.data(), recvcounts.data(), rdispls.data(),
        MPI_COMM_WORLD);

    // Build ghost node mapping
    growt::StaticGhostNodeMapping old_to_new_mapping(rdispls.back() + 1);
    for (const auto& [old_global_label, new_global_label]: recvbuf) {
        // 0 cannot be used as key
        old_to_new_mapping.insert(old_global_label + 1, new_global_label);
    }

    return old_to_new_mapping;
}

DistributedGraph build_graph_sorted(EdgeList edge_list, scalable_vector<GlobalNodeID> node_distribution) {
    const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
    const GlobalNodeID from = node_distribution[rank];
    const GlobalNodeID to   = node_distribution[rank + 1];
    KASSERT(from <= to, "invalid node range", assert::light);

    const NodeID n = static_cast<NodeID>(to - from);
    const EdgeID m = static_cast<EdgeID>(edge_list.size());

    // Sort edges by source / target nodes
    if (!std::is_sorted(edge_list.begin(), edge_list.end())) {
        std::sort(edge_list.begin(), edge_list.end());
    }

    // Count node degrees
    scalable_vector<EdgeID> degrees(n + 1);
    for (const auto& [global_u, global_v]: edge_list) {
        // u should be assigned to this PE
        KASSERT(from <= global_u);
        KASSERT(global_u < to);

        const NodeID local_u = static_cast<NodeID>(global_u - from);
        ++degrees[local_u + 1];
    }

    // Sort nodes by degree bucket
    shm::parallel::prefix_sum(
        degrees.begin(), degrees.end(), degrees.begin()); // sort_by_degree_buckets expects the prefix sum
    const auto  node_permutation       = shm::graph::sort_by_degree_buckets<scalable_vector, false>(degrees);
    const auto& permutation_old_to_new = node_permutation.old_to_new;
    const auto& permutation_new_to_old = node_permutation.new_to_old;

    // Exchange new labels for ghost nodes
    const auto permutation_ghost_old_to_new = remap_ghost_nodes(edge_list, permutation_old_to_new, node_distribution);

    // Build graph data structure
    scalable_vector<EdgeID> nodes(n + 1);
    scalable_vector<NodeID> edges(m);

    // --> Nodes
    tbb::parallel_for<NodeID>(0, n, [&](const NodeID new_u) {
        const NodeID old_u = permutation_new_to_old[new_u];
        nodes[new_u + 1]   = degrees[old_u + 1] - degrees[old_u]; // Node degree
    });
    shm::parallel::prefix_sum(nodes.begin(), nodes.end(), nodes.begin());

    // --> Edges
    DBG << V(node_distribution);
    graph::GhostNodeMapper ghost_node_mapper(node_distribution);
    tbb::parallel_for<NodeID>(0, n, [&](const NodeID new_u) {
        const NodeID old_u = permutation_new_to_old[new_u];

        for (EdgeID offset_e = 0; offset_e < nodes[new_u + 1] - nodes[new_u]; ++offset_e) {
            const EdgeID new_e = offset_e + nodes[new_u];
            const EdgeID old_e = offset_e + degrees[old_u];

            const auto& [edge_u, edge_v] = edge_list[old_e];
            KASSERT(edge_u - from == old_u, V(edge_u) << V(from));

            if (from <= edge_v && edge_v < to) {
                edges[new_e] = permutation_old_to_new[edge_v - from];
            } else {
                auto edge_v_it = permutation_ghost_old_to_new.find(edge_v + 1);
                KASSERT(edge_v_it != permutation_ghost_old_to_new.end());
                const GlobalNodeID remapped_edge_v = (*edge_v_it).second;
                KASSERT(
                    remapped_edge_v < from || remapped_edge_v >= to, V(from)
                                                                         << V(remapped_edge_v) << V(to) << V(edge_v));
                edges[new_e] = ghost_node_mapper.new_ghost_node(remapped_edge_v);
            }
        }
    });

    auto mapped_ghost_nodes = ghost_node_mapper.finalize();

    DistributedGraph graph{
        std::move(node_distribution),
        mpi::build_distribution_from_local_count<GlobalEdgeID, scalable_vector>(m, MPI_COMM_WORLD),
        std::move(nodes),
        std::move(edges),
        std::move(mapped_ghost_nodes.ghost_owner),
        std::move(mapped_ghost_nodes.ghost_to_global),
        std::move(mapped_ghost_nodes.global_to_ghost),
        true,
        MPI_COMM_WORLD};
    KASSERT(graph::debug::validate(graph), "", assert::heavy);
    return graph;
}

DistributedGraph build_graph(const EdgeList& edge_list, scalable_vector<GlobalNodeID> node_distribution) {
    SCOPED_TIMER("Build graph from edge list");

    const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
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
    const auto [size, rank] = mpi::get_comm_info(MPI_COMM_WORLD);
    const GlobalNodeID to   = range.second;

    scalable_vector<GlobalNodeID> node_distribution(size + 1);
    mpi::allgather(&to, 1, node_distribution.data() + 1, 1, MPI_COMM_WORLD);
    return node_distribution;
}

KaGen create_generator_object(const GeneratorContext ctx) {
    KaGen gen(MPI_COMM_WORLD);
    gen.SetSeed(ctx.seed);
    if (ctx.validate_graph) {
        gen.EnableUndirectedGraphVerification();
    }
    gen.EnableOutput(true);
    gen.EnableBasicStatistics();
    return gen;
}

KaGenResult create_gnm(const GeneratorContext ctx) {
    const GlobalNodeID n = (static_cast<GlobalNodeID>(1) << ctx.n) * ctx.scale;
    const GlobalEdgeID m = (static_cast<GlobalEdgeID>(1) << ctx.m) * ctx.scale;

    LOG << "Generating GNM(n=" << n << ", m=" << m << ")";
    return create_generator_object(ctx).GenerateUndirectedGNM(n, m);
}

KaGenResult create_rgg2d(const GeneratorContext ctx) {
    const GlobalNodeID n = (static_cast<GlobalNodeID>(1) << ctx.n) * ctx.scale;
    const GlobalEdgeID m = (static_cast<GlobalEdgeID>(1) << ctx.m) * ctx.scale;

    LOG << "Generating RGG2D(n=" << n << ", m=" << m << ")";
    return create_generator_object(ctx).GenerateRGG2D_NM(n, m);
}

KaGenResult create_rgg3d(const GeneratorContext ctx) {
    const GlobalNodeID n = (static_cast<GlobalNodeID>(1) << ctx.n) * ctx.scale;
    const GlobalEdgeID m = (static_cast<GlobalEdgeID>(1) << ctx.m) * ctx.scale;

    LOG << "Generating RGG3D(n=" << n << ", m=" << m << ")";
    return create_generator_object(ctx).GenerateRGG3D_NM(n, m);
}

KaGenResult create_rdg2d(const GeneratorContext ctx) {
    const GlobalEdgeID m = (static_cast<GlobalEdgeID>(1) << ctx.m) * ctx.scale;

    LOG << "Generating RDG2D(m=" << m << ", periodic=" << ctx.periodic << ")";
    return create_generator_object(ctx).GenerateRDG2D_M(m, ctx.periodic);
}

KaGenResult create_rdg3d(const GeneratorContext ctx) {
    const GlobalEdgeID m = (static_cast<GlobalEdgeID>(1) << ctx.m) * ctx.scale;

    LOG << "Generating RDG3D(m=" << m << ", periodic=" << ctx.periodic << ")";
    return create_generator_object(ctx).GenerateRDG3D_M(m);
}

KaGenResult create_rhg(const GeneratorContext ctx) {
    const GlobalNodeID n = (static_cast<GlobalNodeID>(1) << ctx.n) * ctx.scale;
    const GlobalEdgeID m = (static_cast<GlobalNodeID>(1) << ctx.m) * ctx.scale;

    LOG << "Generating RHG(gamma=" << ctx.gamma << ", n=" << n << ", m=" << m << ")";
    return create_generator_object(ctx).GenerateRHG_NM(ctx.gamma, n, m);
}

KaGenResult create_grid2d(const GeneratorContext ctx) {
    const GlobalNodeID n = (static_cast<GlobalNodeID>(1) << ctx.n) * ctx.scale;

    LOG << "Generating Grid2D(n=" << n << ", p=" << ctx.p << ")";
    return create_generator_object(ctx).GenerateGrid2D_N(n, ctx.p);
}

KaGenResult create_grid3d(const GeneratorContext ctx) {
    const GlobalNodeID n = (static_cast<GlobalNodeID>(1) << ctx.n) * ctx.scale;

    LOG << "Generating Grid3D(n=" << n << ", p=" << ctx.p << ")";
    return create_generator_object(ctx).GenerateGrid3D_N(n, ctx.p);
}

KaGenResult create_rmat(const GeneratorContext ctx) {
    const GlobalNodeID n = (static_cast<GlobalNodeID>(1) << ctx.n) * ctx.scale;
    const GlobalEdgeID m = (static_cast<GlobalEdgeID>(1) << ctx.m) * ctx.scale;

    LOG << "Generating R-MAT(n=" << n << ", m=" << m << ", a=" << ctx.prob_a << ", b=" << ctx.prob_b
        << ", c=" << ctx.prob_c << ")";
    return create_generator_object(ctx).GenerateRMAT(n, m, ctx.prob_a, ctx.prob_b, ctx.prob_c);
}
} // namespace

DistributedGraph generate(const GeneratorContext ctx) {
    auto [edges, local_range] = [&] {
        switch (ctx.type) {
            case GeneratorType::GNM:
                return create_gnm(ctx);

            case GeneratorType::RGG2D:
                return create_rgg2d(ctx);

            case GeneratorType::RGG3D:
                return create_rgg3d(ctx);

            case GeneratorType::RHG:
                return create_rhg(ctx);

            case GeneratorType::RDG2D:
                return create_rdg2d(ctx);

            case GeneratorType::GRID2D:
                return create_grid2d(ctx);

            case GeneratorType::GRID3D:
                return create_grid3d(ctx);

            case GeneratorType::RMAT:
                return create_rmat(ctx);

            default:
                FATAL_ERROR << "selected graph generator is not implemented";
        }

        __builtin_unreachable();
    }();

    return build_graph_sorted(std::move(edges), build_node_distribution(local_range));
}
} // namespace dkaminpar::graphgen
