/*******************************************************************************
 * @file:   dbalancing_benchmark.cc
 * @author: Daniel Seemaier
 * @date:   12.04.2022
 * @brief:  Benchmark for distributed graph partition balancing.
 ******************************************************************************/
// This must come first since it redefines output macros (LOG DBG etc)
// clang-format off
#include "dkaminpar/definitions.h"
// clang-format on

#include <fstream>

#include <mpi.h>

#include "dkaminpar/coarsening/global_clustering_contraction.h"
#include "dkaminpar/coarsening/locking_label_propagation_clustering.h"
#include "dkaminpar/context.h"
#include "dkaminpar/io.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/mpi/graph_communication.h"
#include "dkaminpar/refinement/balancer.h"

#include "kaminpar/application/arguments.h"
#include "kaminpar/definitions.h"

#include "common/logger.h"
#include "common/random.h"
#include "common/timer.h"

#include "apps/apps.h"
#include "apps/dkaminpar/arguments.h"

using namespace kaminpar;
using namespace kaminpar::dist;

int main(int argc, char* argv[]) {
    Context     ctx = create_default_context();
    std::string partition_filename;

    { // init MPI
        int provided_thread_support;
        MPI_Init_thread(&argc, &argv, ctx.parallel.mpi_thread_support, &provided_thread_support);
        if (provided_thread_support != ctx.parallel.mpi_thread_support) {
            LOG_WARNING << "Desired MPI thread support unavailable: set to " << provided_thread_support;
            if (provided_thread_support == MPI_THREAD_SINGLE) {
                if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
                    LOG_ERROR << "Your MPI library does not support multithreading. This might cause malfunction.";
                }
                provided_thread_support = MPI_THREAD_FUNNELED; // fake multithreading level for application
            }
            ctx.parallel.mpi_thread_support = provided_thread_support;
        }
    }

    Arguments args;
    args.positional()
        .argument("graph", "Input graph", &ctx.graph_filename)
        .argument("partition", "Input partition", &partition_filename);
    args.group("Partition").argument("epsilon", "Max. partition imbalance", &ctx.partition.epsilon, 'e');
    args.group("Misc")
        .argument("threads", "Number of threads", &ctx.parallel.num_threads, 't')
        .argument("seed", "Seed for RNG", &ctx.seed, 's');
    create_balancing_options(ctx.refinement.balancing, args, "", "b");
    args.parse(argc, argv);

    print_identifier(argc, argv);
    LOG << "MPI size=" << mpi::get_comm_size(MPI_COMM_WORLD);

    // Initialize random number generator
    Random::seed = ctx.seed;

    // Initialize TBB
    init_numa();
    auto gc = init_parallelism(ctx.parallel.num_threads);

    // Load graph
    const auto graph = TIMED_SCOPE("IO") {
        auto graph = io::metis::read_node_balanced(ctx.graph_filename);
        mpi::barrier(MPI_COMM_WORLD);
        return graph;
    };
    LOG << "Loaded graph with n=" << graph.global_n() << " m=" << graph.global_m();
    KASSERT(graph::debug::validate(graph));

    // Load partition
    auto partition = io::partition::read<scalable_vector<parallel::Atomic<BlockID>>>(partition_filename, graph.n());
    KASSERT(partition.size() == graph.n(), "", assert::always);

    // Communicate blocks of ghost nodes
    for (NodeID u = graph.n(); u < graph.total_n(); ++u) {
        partition.push_back(0);
    }

    struct Message {
        NodeID  node;
        BlockID block;
    };

    mpi::graph::sparse_alltoall_interface_to_pe<Message>(
        graph,
        [&](const NodeID u) -> Message {
            return {u, partition[u]};
        },
        [&](const auto buffer, const PEID pe) {
            tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
                const auto& [local_node_on_other_pe, block] = buffer[i];
                const NodeID local_node = graph.global_to_local_node(graph.offset_n(pe) + local_node_on_other_pe);
                partition[local_node]   = block;
            });
        }
    );

    // Create partitioned graph object
    const BlockID local_k = *std::max_element(partition.begin(), partition.end()) + 1;
    const BlockID k       = mpi::allreduce(local_k, MPI_MAX, MPI_COMM_WORLD);

    scalable_vector<BlockWeight> local_block_weights(k);
    for (const NodeID u: graph.nodes()) {
        local_block_weights[partition[u]] += graph.node_weight(u);
    }

    scalable_vector<BlockWeight> global_block_weight_nonatomic(k);
    mpi::allreduce(
        local_block_weights.data(), global_block_weight_nonatomic.data(), static_cast<int>(k), MPI_SUM, MPI_COMM_WORLD
    );

    scalable_vector<parallel::Atomic<BlockWeight>> block_weights(k);
    std::copy(global_block_weight_nonatomic.begin(), global_block_weight_nonatomic.end(), block_weights.begin());

    DistributedPartitionedGraph p_graph(&graph, k, std::move(partition), std::move(block_weights));

    // Setup context
    ctx.partition.k = k;
    ctx.setup(graph);

    // Output statistics
    const auto cut_before       = metrics::edge_cut(p_graph);
    const auto imbalance_before = metrics::imbalance(p_graph);
    LOG << "PARTITION k=" << k << " cut=" << cut_before << " imbalance=" << imbalance_before;

    // Run balancer
    START_TIMER("Balancer");
    START_TIMER("Allocation");
    DistributedBalancer balancer(ctx);
    STOP_TIMER();
    START_TIMER("Initialization");
    balancer.initialize(p_graph);
    STOP_TIMER();
    START_TIMER("Balancing");
    balancer.balance(p_graph, ctx.partition);
    STOP_TIMER();
    STOP_TIMER();

    // Output statistics
    const auto cut_after       = metrics::edge_cut(p_graph);
    const auto imbalance_after = metrics::imbalance(p_graph);
    LOG << "RESULT cut=" << cut_after << " imbalance=" << imbalance_after;
    mpi::barrier(MPI_COMM_WORLD);

    LOG << p_graph.block_weights();

    if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0 && !ctx.quiet) {
        Timer::global().print_machine_readable(std::cout);
    }
    LOG;
    if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0 && !ctx.quiet) {
        Timer::global().print_human_readable(std::cout);
    }
    LOG;

    MPI_Finalize();
    return 0;
}
