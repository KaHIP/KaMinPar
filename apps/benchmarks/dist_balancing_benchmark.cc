/*******************************************************************************
 * @file:   dist_balancing_benchmark.cc
 * @author: Daniel Seemaier
 * @date:   12.04.2022
 * @brief:  Benchmark for the distributed balancing algorithm.
 ******************************************************************************/
// clang-format off
#include "common/CLI11.h"
#include "dkaminpar/definitions.h"
// clang-format on

#include <fstream>

#include <mpi.h>

#include "dkaminpar/arguments.h"
#include "dkaminpar/coarsening/global_clustering_contraction.h"
#include "dkaminpar/coarsening/locking_label_propagation_clustering.h"
#include "dkaminpar/context.h"
#include "dkaminpar/factories.h"
#include "dkaminpar/io.h"
#include "dkaminpar/metrics.h"
#include "dkaminpar/mpi/graph_communication.h"
#include "dkaminpar/presets.h"

#include "kaminpar/definitions.h"

#include "common/logger.h"
#include "common/random.h"
#include "common/timer.h"

#include "apps/benchmarks/dist_benchmarks_common.h"
#include "apps/mpi_apps.h"

using namespace kaminpar;
using namespace kaminpar::dist;

int main(int argc, char* argv[]) {
    init_mpi(argc, argv);

    Context     ctx = create_default_context();
    std::string graph_filename;
    std::string partition_filename;

    // Change default to only balancer, no other refiner
    ctx.refinement.algorithms = {KWayRefinementAlgorithm::GREEDY_BALANCER};

    CLI::App app;
    app.add_option("-G", graph_filename);
    app.add_option("-P", partition_filename);
    app.add_option("-e", ctx.partition.epsilon);
    app.add_option("-t", ctx.parallel.num_threads);
    create_greedy_balancer_options(&app, ctx);
    CLI11_PARSE(app, argc, argv);

    auto gc = init(ctx, argc, argv);

    auto graph      = load_graph(graph_filename);
    auto p_graph    = load_graph_partition(graph, partition_filename);
    ctx.partition.k = p_graph.k();
    ctx.setup(graph);

    auto balancer = factory::create_refinement_algorithm(ctx);

    TIMED_SCOPE("Balancer") {
        TIMED_SCOPE("Initialization") {
            balancer->initialize(graph);
        };
        TIMED_SCOPE("Balancing") {
            balancer->refine(p_graph, ctx.partition);
        };
    };

    const auto cut_after       = metrics::edge_cut(p_graph);
    const auto imbalance_after = metrics::imbalance(p_graph);
    LOG << "RESULT cut=" << cut_after << " imbalance=" << imbalance_after;
    mpi::barrier(MPI_COMM_WORLD);

    if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
        Timer::global().print_machine_readable(std::cout);
    }
    if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
        Timer::global().print_human_readable(std::cout);
    }

    return MPI_Finalize();
}
