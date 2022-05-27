/*******************************************************************************
 * @file:   dcontraction_benchmark.cc
 *
 * @author: Daniel Seemaier
 * @date:   27.05.2022
 * @brief:  Benchmark for distributed graph contraction.
 ******************************************************************************/
// This must come first since it redefines output macros (LOG DBG etc)
// clang-format off
#include "application/arguments_parser.h"
#include "datastructure/distributed_graph.h"
#include "dkaminpar/definitions.h"
// clang-format on

#include <fstream>

#include <mpi.h>

#include "apps/apps.h"
#include "apps/dkaminpar_arguments.h"
#include "dkaminpar/coarsening/global_clustering_contraction.h"
#include "dkaminpar/context.h"
#include "dkaminpar/distributed_io.h"
#include "dkaminpar/graphutils/rearrange_graph.h"
#include "dkaminpar/mpi_wrapper.h"
#include "kaminpar/definitions.h"
#include "kaminpar/utils/logger.h"
#include "kaminpar/utils/random.h"
#include "kaminpar/utils/timer.h"

using namespace dkaminpar;
namespace shm = kaminpar;

void init_mpi(int& argc, char**& argv) {
    int provided_thread_support;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_thread_support);
    if (provided_thread_support != MPI_THREAD_FUNNELED) {
        LOG_WARNING << "Desired MPI thread support unavailable: set to " << provided_thread_support;
        if (provided_thread_support == MPI_THREAD_SINGLE) {
            if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
                LOG_ERROR << "Your MPI library does not support multithreading. This might cause malfunction.";
            }
        }
    }
}

DistributedGraph load_graph(Context& ctx) {
    auto graph = TIMED_SCOPE("IO") {
        auto graph = io::metis::read_node_balanced(ctx.graph_filename);
        KASSERT(graph::debug::validate(graph), "bad input graph", assert::heavy);
        return graph;
    };

    graph = graph::sort_by_degree_buckets(std::move(graph));
    KASSERT(graph::debug::validate(graph), "bad sorted graph", assert::heavy);

    LOG << "Input graph:";
    graph::print_summary(graph);

    ctx.setup(graph);
    return graph;
}

auto load_clustering(Context& ctx, const std::string& filename) {
    return io::partition::read<scalable_vector<Atomic<GlobalNodeID>>>(filename, ctx.partition.local_n());
}

int main(int argc, char* argv[]) {
    init_mpi(argc, argv);

    Context     ctx = create_default_context();
    std::string clustering_filename;

    // Parse command line arguments
    try {
        shm::Arguments args;
        args.positional()
            .argument("graph", "Graph", &ctx.graph_filename)
            .argument("clustering", "Clustering filename", &clustering_filename);
        args.parse(argc, argv);
    } catch (const std::runtime_error& e) {
        std::cout << e.what() << std::endl;
    }
    shm::Logger::set_quiet_mode(ctx.quiet);

    shm::print_identifier(argc, argv);
    LOG << "MPI size=" << mpi::get_comm_size(MPI_COMM_WORLD);
    LOG << "CONTEXT " << ctx;

    // Initialize random number generator
    shm::Randomize::seed = ctx.seed;

    // Initialize TBB
    auto gc = shm::init_parallelism(ctx.parallel.num_threads);
    if (ctx.parallel.use_interleaved_numa_allocation) {
        shm::init_numa();
    }

    // Load data
    const auto graph      = load_graph(ctx);
    const auto clustering = load_clustering(ctx, clustering_filename);

    // Compute coarse graph
    START_TIMER("Contraction");
    const auto [c_graph, c_mappuing] =
        coarsening::contract_global_clustering(graph, clustering, ctx.coarsening.global_contraction_algorithm);
    STOP_TIMER();

    LOG << "Coarse graph:";
    graph::print_summary(c_graph);

    // Output statistics
    mpi::barrier();
    if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0 && !ctx.quiet) {
        shm::Timer::global().print_machine_readable(std::cout);
    }
    LOG;
    if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0 && !ctx.quiet) {
        shm::Timer::global().print_human_readable(std::cout);
    }
    LOG;

    MPI_Finalize();
    return 0;
}
