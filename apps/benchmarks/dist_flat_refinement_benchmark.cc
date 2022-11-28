// clang-format off
#include "common/CLI11.h"
// clang-format on

#include <mpi.h>
#include <omp.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "dkaminpar/algorithms/greedy_node_coloring.h"
#include "dkaminpar/arguments.h"
#include "dkaminpar/factories.h"
#include "dkaminpar/graphutils/graph_rearrangement.h"
#include "dkaminpar/io.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/presets.h"
#include "dkaminpar/timer.h"

#include "common/logger.h"
#include "common/timer.h"

#include "apps/apps.h"
#include "apps/mpi_apps.h"

using namespace kaminpar;
using namespace kaminpar::dist;

int main(int argc, char* argv[]) {
    init_mpi(argc, argv);
    const PEID rank = mpi::get_comm_rank(MPI_COMM_WORLD);

    /*****
     * Parse command line arguments
     */
    std::string graph_filename     = "";
    std::string partition_filename = "";
    Context     ctx                = create_default_context();
    CLI::App    app("Distributed Flat Refinement Benchmark");
    app.add_option("-G,--graph", graph_filename, "Input graph")->check(CLI::ExistingFile)->required();
    app.add_option("-P,--partition", partition_filename, "Partition filename")->check(CLI::ExistingFile)->required();
    app.add_option("-k,--k", ctx.partition.k, "Number of blocks")->required();
    create_all_options(&app, ctx);
    CLI11_PARSE(app, argc, argv);

    auto gc = init_parallelism(1);
    omp_set_num_threads(1);
    init_numa();

    /*****
     * Load graph and rearrange graph
     */
    LOG << "Reading graph from " << graph_filename << " ...";
    auto graph = dist::io::read_graph(graph_filename, dist::io::DistributionType::NODE_BALANCED);
    ctx.setup(graph);
    LOG << "Loaded graph with " << graph.global_n() << " nodes and " << graph.global_m() << " edges";
    graph = graph::rearrange(std::move(graph), ctx);

    LOG << "Reading partition from " << partition_filename << " ...";
    DistributedPartitionedGraph p_graph = dist::io::partition::read(partition_filename, graph, ctx.partition.k);

    const EdgeWeight cut_before       = metrics::edge_cut(p_graph);
    const double     imbalance_before = metrics::imbalance(p_graph);
    const bool       feasible_before  = metrics::is_feasible(p_graph, ctx.partition);
    LOG << "Before refinement:";
    LOG << "  Edge cut:  " << cut_before;
    LOG << "  Imbalance: " << imbalance_before;
    LOG << "  Feasible:  " << (feasible_before ? "yes" : "no");

    /****
     * Run label propagation
     */
    auto refiner = factory::create_refinement_algorithm(ctx);
    refiner->initialize(graph);
    refiner->refine(p_graph, ctx.partition);

    const EdgeWeight cut_after       = metrics::edge_cut(p_graph);
    const double     imbalance_after = metrics::imbalance(p_graph);
    const bool       feasible_after  = metrics::is_feasible(p_graph, ctx.partition);
    LOG << "After refinement:";
    LOG << "  Edge cut:  " << cut_after;
    LOG << "  Imbalance: " << imbalance_after;
    LOG << "  Feasible:  " << (feasible_after ? "yes" : "no");

    /*****
     * Clean up and print timer tree
     */
    mpi::barrier(MPI_COMM_WORLD);
    STOP_TIMER();
    finalize_distributed_timer(GLOBAL_TIMER);
    if (rank == 0) {
        Timer::global().print_human_readable(std::cout);
    }
    return MPI_Finalize();
}

