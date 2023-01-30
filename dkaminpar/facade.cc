#include <utility>

#include <mpi.h>
#include <omp.h>
#include <tbb/global_control.h>
#include <tbb/parallel_invoke.h>

#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/datastructures/distributed_graph_builder.h"
#include "dkaminpar/dkaminpar.h"
#include "dkaminpar/factories.h"
#include "dkaminpar/graphutils/graph_rearrangement.h"

#include "kaminpar/context.h"

#include "common/random.h"

namespace kaminpar::dist {
GraphPtr::GraphPtr(std::unique_ptr<DistributedGraph> graph) : ptr(std::move(graph)) {}

GraphPtr::GraphPtr(GraphPtr&&) noexcept            = default;
GraphPtr& GraphPtr::operator=(GraphPtr&&) noexcept = default;

GraphPtr::~GraphPtr() = default;

GraphPtr import_graph(
    MPI_Comm comm, GlobalNodeID* vtxdist, GlobalEdgeID* xadj, GlobalNodeID* adjncy, GlobalNodeWeight* vwgt,
    GlobalEdgeWeight* adjwgt
) {
    SCOPED_TIMER("IO");

    const PEID size = mpi::get_comm_size(comm);
    const PEID rank = mpi::get_comm_rank(comm);

    const NodeID       n    = static_cast<NodeID>(vtxdist[rank + 1] - vtxdist[rank]);
    const GlobalNodeID from = vtxdist[rank];
    const GlobalNodeID to   = vtxdist[rank + 1];
    const EdgeID       m    = static_cast<EdgeID>(xadj[n]);

    scalable_vector<GlobalNodeID> node_distribution(vtxdist, vtxdist + size + 1);
    scalable_vector<GlobalEdgeID> edge_distribution(size + 1);
    edge_distribution[rank] = m;
    MPI_Allgather(
        MPI_IN_PLACE, 1, mpi::type::get<GlobalEdgeID>(), edge_distribution.data(), 1, mpi::type::get<GlobalEdgeID>(),
        comm
    );

    scalable_vector<EdgeID>     nodes;
    scalable_vector<NodeID>     edges;
    scalable_vector<NodeWeight> node_weights;
    scalable_vector<EdgeWeight> edge_weights;
    graph::GhostNodeMapper      mapper(comm, node_distribution);

    tbb::parallel_invoke(
        [&] {
            nodes.resize(n + 1);
            tbb::parallel_for<NodeID>(0, n + 1, [&](const NodeID u) { nodes[u] = xadj[u]; });
        },
        [&] {
            edges.resize(m);
            tbb::parallel_for<EdgeID>(0, m, [&](const EdgeID e) {
                const GlobalNodeID v = adjncy[e];
                if (v >= from && v < to) {
                    edges[e] = static_cast<NodeID>(v - from);
                } else {
                    edges[e] = mapper.new_ghost_node(v);
                }
            });
        },
        [&] {
            if (vwgt == nullptr) {
                return;
            }
            node_weights.resize(n);
            tbb::parallel_for<NodeID>(0, n, [&](const NodeID u) { node_weights[u] = vwgt[u]; });
        },
        [&] {
            if (adjwgt == nullptr) {
                return;
            }
            edge_weights.resize(m);
            tbb::parallel_for<EdgeID>(0, m, [&](const EdgeID e) { edge_weights[e] = adjwgt[e]; });
        }
    );

    auto [global_to_ghost, ghost_to_global, ghost_owner] = mapper.finalize();

    return GraphPtr(std::make_unique<DistributedGraph>(
        std::move(node_distribution), std::move(edge_distribution), std::move(nodes), std::move(edges),
        std::move(node_weights), std::move(edge_weights), std::move(ghost_owner), std::move(ghost_to_global),
        std::move(global_to_ghost), false, comm
    ));
}

std::vector<BlockID>
partition(GraphPtr graph_ptr, Context ctx, const int num_threads, const int seed, const OutputLevel output_level) {
    auto& graph = *graph_ptr.ptr;

    Random::seed = seed;

    ctx.parallel.num_threads = num_threads;
    auto gc = tbb::global_control{tbb::global_control::max_allowed_parallelism, ctx.parallel.num_threads};
    omp_set_num_threads(static_cast<int>(ctx.parallel.num_threads));
    ctx.initial_partitioning.kaminpar->parallel.num_threads = ctx.parallel.num_threads;

    START_TIMER("Partitioning");
    graph        = graph::rearrange(std::move(graph), ctx);
    auto p_graph = factory::create_partitioner(ctx, graph)->partition();
    STOP_TIMER();

    START_TIMER("IO");
    std::vector<BlockID> partition(p_graph.n());
    if (graph.permuted()) {
        tbb::parallel_for<NodeID>(0, p_graph.n(), [&](const NodeID u) {
            partition[u] = p_graph.block(graph.map_original_node(u));
        });
    } else {
        tbb::parallel_for<NodeID>(0, p_graph.n(), [&](const NodeID u) { partition[u] = p_graph.block(u); });
    }
    STOP_TIMER();

    return partition;
}
} // namespace kaminpar::dist
