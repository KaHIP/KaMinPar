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
#include "dkaminpar/metrics.h"

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

namespace {
void print_result_statistics(const DistributedPartitionedGraph& p_graph, const Context& ctx, const bool parseable) {
    // Aggregate timers to display min, max, avg and sd across PEs
    // Disabled: this function requires the same timer hierarchy on all PEs;
    // in deep MGP, this is not always the case
    // if (!ctx.quiet) {
    // finalize_distributed_timer(GLOBAL_TIMER);
    //}

    const bool root = mpi::get_comm_rank(MPI_COMM_WORLD) == 0;

    const auto edge_cut  = metrics::edge_cut(p_graph);
    const auto imbalance = metrics::imbalance(p_graph);
    const auto feasible  = metrics::is_feasible(p_graph, ctx.partition);

    if (root && parseable) {
        LOG << "RESULT cut=" << edge_cut << " imbalance=" << imbalance << " feasible=" << feasible
            << " k=" << p_graph.k();
        std::cout << "TIME ";
        Timer::global().print_machine_readable(std::cout);
    }
    LOG;

    if (root) {
        Timer::global().print_human_readable(std::cout);
    }
    LOG;
    LOG << "-> k=" << p_graph.k();
    LOG << "-> cut=" << edge_cut;
    LOG << "-> imbalance=" << imbalance;
    LOG << "-> feasible=" << feasible;
    if (p_graph.k() <= 512) {
        LOG << "-> block_weights:";
        LOG << logger::TABLE << p_graph.block_weights();
    }

    if (root && (p_graph.k() != ctx.partition.k || !feasible)) {
        LOG_ERROR << "*** Partition is infeasible!";
    }
}
void print_parsable_summary(const Context& ctx, const DistributedGraph& graph, const bool root) {
    if (root) {
        cio::print_delimiter(std::cout);
    }
    LOG << "MPI size=" << ctx.parallel.num_mpis;
    LLOG << "CONTEXT ";
    if (root) {
        print_compact(ctx, std::cout, "");
    }

    const auto n_str       = mpi::gather_statistics_str<GlobalNodeID>(graph.n(), MPI_COMM_WORLD);
    const auto m_str       = mpi::gather_statistics_str<GlobalEdgeID>(graph.m(), MPI_COMM_WORLD);
    const auto ghost_n_str = mpi::gather_statistics_str<GlobalNodeID>(graph.ghost_n(), MPI_COMM_WORLD);
    LOG << "GRAPH "
        << "global_n=" << graph.global_n() << " "
        << "global_m=" << graph.global_m() << " "
        << "n=[" << n_str << "] "
        << "m=[" << m_str << "] "
        << "ghost_n=[" << ghost_n_str << "]";
}

void print_execution_mode(const Context& ctx) {
    LOG << "Execution mode:               " << ctx.parallel.num_mpis << " MPI process"
        << (ctx.parallel.num_mpis > 1 ? "es" : "") << " a " << ctx.parallel.num_threads << " thread"
        << (ctx.parallel.num_threads > 1 ? "s" : "");
    cio::print_delimiter();
}
} // namespace

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

    KASSERT(
        graph::debug::validate_partition(p_graph), "graph partition verification failed after partitioning",
        assert::heavy
    );

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

    mpi::barrier(MPI_COMM_WORLD);
    STOP_TIMER(); // stop root timer
    print_result_statistics(p_graph, app);

    if (output_level == OutputLevel::SILENT || output_level == OutputLevel::FULL) {
        print_result_statistics(p_graph, ctx, output_level == OutputLevel::EXPERIMENT);
    }

    return partition;
}
} // namespace kaminpar::dist
