/*******************************************************************************
 * @file:   dist_coarsening_benchmark.cc
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Benchmark for the distributed coarsening algorithm.
 ******************************************************************************/
// clang-format off
#include "apps/benchmarks/dist_benchmarks_common.h"
// clang-format on

#include <fstream>

#include <mpi.h>

#include "dkaminpar/arguments.h"
#include "dkaminpar/coarsening/global_clustering_contraction.h"
#include "dkaminpar/coarsening/locking_label_propagation_clustering.h"
#include "dkaminpar/context.h"
#include "dkaminpar/presets.h"

#include "kaminpar/definitions.h"

#include "common/logger.h"
#include "common/random.h"
#include "common/timer.h"

#include "apps/apps.h"
#include "apps/mpi_apps.h"

using namespace kaminpar;
using namespace kaminpar::dist;

int main(int argc, char* argv[]) {
    init_mpi(argc, argv);

    Context     ctx = create_default_context();
    std::string graph_filename;

    CLI::App app;
    app.add_option("-G", graph_filename);
    app.add_option("-t", ctx.parallel.num_threads);
    create_coarsening_options(&app, ctx);
    CLI11_PARSE(app, argc, argv);

    auto gc = init(ctx, argc, argv);

    auto graph = load_graph(graph_filename);
    ctx.setup(graph);

    std::vector<dist::DistributedGraph> graph_hierarchy;

    const dist::DistributedGraph* c_graph = &graph;
    while (c_graph->global_n() > ctx.partition.k * ctx.coarsening.contraction_limit) {
        const auto max_cluster_weight = shm::compute_max_cluster_weight(
            c_graph->global_n(), c_graph->total_node_weight(), ctx.initial_partitioning.kaminpar.partition,
            ctx.initial_partitioning.kaminpar.coarsening
        );
        LOG << "... computing clustering";

        START_TIMER("Clustering Algorithm", "Level " + std::to_string(graph_hierarchy.size()));
        dist::LockingLabelPropagationClustering clustering_algorithm(ctx);
        auto& clustering = clustering_algorithm.compute_clustering(*c_graph, max_cluster_weight);
        STOP_TIMER();

        LOG << "... contracting";

        START_TIMER("Contraction", "Level " + std::to_string(graph_hierarchy.size()));
        auto result = contract_global_clustering(*c_graph, clustering, ctx.coarsening.global_contraction_algorithm);
        STOP_TIMER();
        dist::graph::debug::validate(contracted_graph);

        const bool converged = contracted_graph.global_n() == c_graph->global_n();
        graph_hierarchy.push_back(std::move(contracted_graph));
        c_graph = &graph_hierarchy.back();

        LOG << "=> n=" << c_graph->global_n() << " m=" << c_graph->global_m()
            << " max_node_weight=" << c_graph->max_node_weight() << " max_cluster_weight=" << max_cluster_weight;
        SLOG << "n=" << c_graph->n() << " total_n=" << c_graph->total_n() << " ghost_n=" << c_graph->ghost_n()
             << " m=" << c_graph->m();
        if (converged) {
            LOG << "==> Coarsening converged";
            break;
        }
    }

    mpi::barrier(MPI_COMM_WORLD);
    if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
        Timer::global().print_machine_readable(std::cout);
    }
    if (mpi::get_comm_rank(MPI_COMM_WORLD) == 0) {
        Timer::global().print_human_readable(std::cout);
    }

    return MPI_Finalize();
}
