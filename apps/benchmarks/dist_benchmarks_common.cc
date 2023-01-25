/*******************************************************************************
 * @file:   dist_benchmarks_common.cc
 * @author: Daniel Seemaier
 * @date:   25.01.2023
 * @brief:  Common functions for distributed benchmarks.
 ******************************************************************************/
#include "apps/benchmarks/dist_benchmarks_common.h"

#include <kassert/kassert.hpp>
#include <omp.h>

#include "dkaminpar/graphutils/graph_synchronization.h"
#include "dkaminpar/io.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/mpi/graph_communication.h"

#include "common/assertion_levels.h"
#include "common/logger.h"
#include "common/random.h"
#include "common/timer.h"

#include "apps/apps.h"

namespace kaminpar::dist {
tbb::global_control init(const Context& ctx, int& argc, char**& argv) {
    Random::seed = ctx.seed;

    print_identifier(argc, argv);
    LOG << "MPI size=" << mpi::get_comm_size(MPI_COMM_WORLD);

    omp_set_num_threads(static_cast<int>(ctx.parallel.num_threads));
    if (ctx.parallel.use_interleaved_numa_allocation) {
        init_numa();
    }
    return init_parallelism(ctx.parallel.num_threads);
}

DistributedGraph load_graph(const std::string& filename) {
    auto graph = dist::io::metis::read_node_balanced(filename);
    KASSERT(graph::debug::validate(graph), "bad input graph", assert::heavy);

    LOG << "Input graph:";
    graph::print_summary(graph);

    return graph;
}

DistributedPartitionedGraph load_graph_partition(const DistributedGraph& graph, const std::string& filename) {
    auto partition = dist::io::partition::read<scalable_vector<BlockID>>(filename, graph.n());

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

    scalable_vector<BlockWeight> block_weights(k);
    for (const NodeID u: graph.nodes()) {
        block_weights[partition[u]] += graph.node_weight(u);
    }
    MPI_Allreduce(
        MPI_IN_PLACE, block_weights.data(), asserting_cast<int>(k), mpi::type::get<BlockWeight>(), MPI_SUM,
        graph.communicator()
    );

    DistributedPartitionedGraph p_graph(&graph, k, std::move(partition), std::move(block_weights));

    LOG << "Input partition:";
    LOG << "  Edge cut:  " << metrics::edge_cut(p_graph);
    LOG << "  Imbalance: " << metrics::imbalance(p_graph);

    return p_graph;
}
} // namespace kaminpar::dist
