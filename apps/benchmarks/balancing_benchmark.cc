/*******************************************************************************
 * @file:   balancing_benchmark.cc
 *
 * @author: Daniel Seemaier
 * @date:   12.04.2022
 * @brief:  Performance benchmark for shared-memory balancing algorithms.
 ******************************************************************************/
#include "apps/apps.h"
#include "kaminpar/application/arguments.h"
#include "kaminpar/application/arguments_parser.h"
#include "kaminpar/coarsening/cluster_coarsener.h"
#include "kaminpar/coarsening/label_propagation_clustering.h"
#include "kaminpar/context.h"
#include "kaminpar/factories.h"
#include "kaminpar/io.h"
#include "kaminpar/metrics.h"
#include "kaminpar/utils/random.h"
#include "kaminpar/utils/timer.h"

using namespace kaminpar;

int main(int argc, char* argv[]) {
    Context     ctx = create_default_context();
    std::string partition_filename;

    // parse command line arguments
    Arguments args;
    args.positional()
        .argument("graph", "Input graph", &ctx.graph_filename)
        .argument("partition", "Input Partition", &partition_filename);
    app::create_balancer_refinement_context_options(ctx.refinement.balancer, args, "Balancer", "b");
    args.group("Partition").argument("epsilon", "Max. partition imbalance", &ctx.partition.epsilon, 'e');
    args.group("Misc")
        .argument("threads", "Number of threads", &ctx.parallel.num_threads, 't')
        .argument("seed", "Seed for RNG", &ctx.seed, 's');
    args.parse(argc, argv);

    if (!std::ifstream(ctx.graph_filename)) {
        FATAL_ERROR << "Graph file cannot be read. Ensure that the file exists and is readable.";
    }
    if (!std::ifstream(partition_filename)) {
        FATAL_ERROR << "Partition file cannot be read. Ensure that the file exists and is readable.";
    }

    // init components
    init_numa();
    auto gc = init_parallelism(ctx.parallel.num_threads);
    GLOBAL_TIMER.enable(TIMER_BENCHMARK);
    Randomize::seed = ctx.seed;

    // load graph
    START_TIMER("IO");
    Graph graph = io::metis::read(ctx.graph_filename);
    STOP_TIMER();
    LOG << "GRAPH n=" << graph.n() << " m=" << graph.m();

    // load partition
    auto          partition = io::partition::read<StaticArray<BlockID>>(partition_filename);
    const BlockID k         = *std::max_element(partition.begin(), partition.end()) + 1;
    KASSERT(partition.size() == graph.n(), "", assert::always);
    PartitionedGraph p_graph(graph, k, std::move(partition));

    ctx.partition.k = k;
    ctx.setup(graph);

    // output statistics
    const EdgeWeight cut_before       = metrics::edge_cut(p_graph);
    const double     imbalance_before = metrics::imbalance(p_graph);
    LOG << "PARTITION k=" << k << " cut=" << cut_before << " imbalance=" << imbalance_before;

    // run balancer
    START_TIMER("Balancer");
    START_TIMER("Allocation");
    auto balancer = factory::create_balancer(graph, ctx.partition, ctx.refinement);
    STOP_TIMER();
    START_TIMER("Initialization");
    balancer->initialize(p_graph);
    STOP_TIMER();
    START_TIMER("Balancing");
    balancer->balance(p_graph, ctx.partition);
    STOP_TIMER();
    STOP_TIMER();

    // output statistics
    const EdgeWeight cut_after       = metrics::edge_cut(p_graph);
    const double     imbalance_after = metrics::imbalance(p_graph);
    LOG << "RESULT cut=" << cut_after << " imbalance=" << imbalance_after;

    Timer::global().print_machine_readable(std::cout);
    Timer::global().print_human_readable(std::cout);
}
