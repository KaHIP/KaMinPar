/*******************************************************************************
 * @file:   kaminpar.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
// clang-format off
#include "common/CLI11.h"
// clang-format on

#include <iostream>

#include <tbb/parallel_for.h>

#include "context.h"

#include "kaminpar/arguments.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/definitions.h"
#include "kaminpar/graphutils/graph_rearrangement.h"
#include "kaminpar/input_validator.h"
#include "kaminpar/io.h"
#include "kaminpar/metrics.h"
#include "kaminpar/partitioning/partitioning.h"
#include "kaminpar/presets.h"

#include "common/assertion_levels.h"
#include "common/console_io.h"
#include "common/logger.h"
#include "common/random.h"
#include "common/timer.h"

#include "apps/apps.h"
#include "apps/environment.h"

using namespace kaminpar;
using namespace kaminpar::shm;
using namespace std::string_literals;

void print_statistics(const PartitionedGraph& p_graph, const Context& ctx) {
    const EdgeWeight cut       = metrics::edge_cut(p_graph);
    const double     imbalance = metrics::imbalance(p_graph);
    const bool       feasible  = metrics::is_feasible(p_graph, ctx.partition);

    // statistics output that is easy to parse
    if (ctx.parsable_output) {
        if (!ctx.quiet) {
            Timer::global().print_machine_readable(std::cout);
        }
        LOG << "RESULT cut=" << cut << " imbalance=" << imbalance << " feasible=" << feasible << " k=" << p_graph.k();
        LOG;
    }

    // statistics output that is easy to read
    if (!ctx.quiet) {
        Timer::global().print_human_readable(std::cout);
    }
    LOG;
    LOG << "-> k=" << p_graph.k();
    LOG << "-> cut=" << cut;
    LOG << "-> imbalance=" << imbalance;
    LOG << "-> feasible=" << feasible;
    if (p_graph.k() <= 512) {
        LOG << "-> block weights:";
        LOG << logger::TABLE << p_graph.block_weights();
    }
    if (p_graph.k() != ctx.partition.k || !feasible) {
        LOG_ERROR << "*** Partition is infeasible!";
    }
}

std::string generate_partition_filename(const Context& ctx) {
    std::stringstream filename;
    filename << str::extract_basename(ctx.graph_filename);
    filename << "__t" << ctx.parallel.num_threads;
    filename << "__k" << ctx.partition.k;
    filename << "__eps" << ctx.partition.epsilon;
    filename << "__seed" << ctx.seed;
    filename << ".partition";
    return filename.str();
}

Context setup_context(CLI::App& app, int argc, char* argv[]) {
    Context ctx         = create_default_context();
    bool    dump_config = false;

    app.set_config("-C,--config", "", "Read parameters from a TOML configuration file.", false);
    app.add_option_function<std::string>(
           "-P,--preset",
           [&](const std::string preset) {
               if (preset == "default") {
                   ctx = create_default_context();
               } else if (preset == "largek") {
                   ctx = create_largek_context();
               }
           }
    )
        ->check(CLI::IsMember({"default", "largek"}))
        ->description(R"(Use a configuration preset:
  - default: default parameters
  - largek:  reduce repetitions during initial partitioning (better performance if k is large))");

    // Mandatory
    auto* mandatory_group = app.add_option_group("Application")->require_option(1);

    // Mandatory -> either dump config ...
    mandatory_group->add_flag("--dump-config", dump_config)
        ->configurable(false)
        ->description(R"(Print the current configuration and exit.
The output should be stored in a file and can be used by the -C,--config option)");

    // Mandatory -> ... or partition a graph
    auto* gp_group = mandatory_group->add_option_group("Partitioning")->silent(true);
    gp_group->add_option("-G,--graph", ctx.graph_filename, "Input graph in METIS file format.")
        ->configurable(false)
        ->required();
    gp_group->add_option("-k,--k", ctx.partition.k, "Number of blocks in the partition.")
        ->configurable(false)
        ->required();

    // Application options
    app.add_option("-s,--seed", ctx.seed, "Seed for random number generation.")->default_val(ctx.seed);
    app.add_option("-o,--output", ctx.partition_filename, "Name of the partition file.")->configurable(false);
    app.add_option(
           "--output-directory", ctx.partition_directory, "Directory in which the partition file should be placed."
    )
        ->capture_default_str();
    app.add_flag("--degree-weights", ctx.degree_weights, "Use node degrees as node weights.");
    app.add_flag("-q,--quiet", ctx.quiet, "Suppress all console output.");
    app.add_option("-t,--threads", ctx.parallel.num_threads, "Number of threads to be used.")
        ->check(CLI::NonNegativeNumber)
        ->default_val(ctx.parallel.num_threads);
    app.add_flag("-p,--parsable", ctx.parsable_output, "Use an output format that is easier to parse.");
    app.add_flag("--unchecked-io", ctx.unchecked_io, "Run without format checks of the input graph (in Release mode).");
    app.add_flag("--validate-io", ctx.validate_io, "Validate the format of the input graph extensively.");

    // Algorithmic options
    create_all_options(&app, ctx);

    app.parse(argc, argv);

    // Only dump config and exit
    if (dump_config) {
        CLI::App dump;
        create_all_options(&dump, ctx);
        std::cout << dump.config_to_str(true, true);
        std::exit(0);
    }

    if (ctx.partition_filename.empty()) {
        ctx.partition_filename = generate_partition_filename(ctx);
    }

    return ctx;
}

int main(int argc, char* argv[]) {
    CLI::App app("KaMinPar: (Somewhat) Minimal Deep Multilevel Graph Partitioner");
    Context  ctx;
    try {
        ctx = setup_context(app, argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    // Disable console output in quiet mode;
    Logger::set_quiet_mode(ctx.quiet);

    cio::print_kaminpar_banner();
    cio::print_build_identifier<NodeID, EdgeID, NodeWeight, EdgeWeight>(Environment::GIT_SHA1, Environment::HOSTNAME);
    cio::print_delimiter();

    // Initialize
    Random::seed = ctx.seed;
    auto gc      = init_parallelism(ctx.parallel.num_threads); // must stay alive
    if (ctx.parallel.use_interleaved_numa_allocation) {
        init_numa();
    }

    // Load input graph
    const double original_epsilon    = ctx.partition.epsilon;
    bool         need_postprocessing = false;

    auto [graph, permutations] = [&] {
        StaticArray<EdgeID>     nodes;
        StaticArray<NodeID>     edges;
        StaticArray<NodeWeight> node_weights;
        StaticArray<EdgeWeight> edge_weights;

        START_TIMER("IO");
        if (ctx.unchecked_io) {
            shm::io::metis::read<false>(ctx.graph_filename, nodes, edges, node_weights, edge_weights);
        } else {
            shm::io::metis::read<true>(ctx.graph_filename, nodes, edges, node_weights, edge_weights);
        }

        if (ctx.validate_io) {
            validate_undirected_graph(nodes, edges, node_weights, edge_weights);
        }

        if (ctx.degree_weights) {
            if (node_weights.size() != nodes.size() - 1) {
                node_weights.resize_without_init(nodes.size() - 1);
            }
            tbb::parallel_for<NodeID>(0, node_weights.size(), [&node_weights, &nodes](const NodeID u) {
                node_weights[u] = nodes[u + 1] - nodes[u];
            });
        }

        const NodeID n_before_preprocessing = nodes.size();
        STOP_TIMER();

        // sort nodes by degree bucket and rearrange graph, remove isolated nodes
        START_TIMER("Partitioning");
        START_TIMER("Preprocessing");
        auto node_permutations = graph::rearrange_graph(ctx.partition, nodes, edges, node_weights, edge_weights);
        need_postprocessing    = nodes.size() < n_before_preprocessing;
        STOP_TIMER();
        STOP_TIMER();

        return std::pair{
            Graph{std::move(nodes), std::move(edges), std::move(node_weights), std::move(edge_weights), true},
            std::move(node_permutations)};
    }();

    // Setup graph dependent context parameters
    ctx.setup(graph);
    ctx.print(std::cout);

    if (ctx.parsable_output) {
        std::cout << "CONTEXT ";
        ctx.print_compact(std::cout);
        std::cout << "\n";

        LOG << "INPUT graph=" << ctx.graph_filename << " "
            << "n=" << graph.n() << " "
            << "m=" << graph.m() << " "
            << "k=" << ctx.partition.k << " "
            << "epsilon=" << ctx.partition.epsilon << " ";
        LOG << "==> max_block_weight=" << ctx.partition.block_weights.max(0);
    }

    // Perform actual partitioning
    PartitionedGraph p_graph = partitioning::partition(graph, ctx);

    // Re-integrate isolated nodes that were cut off during preprocessing
    if (need_postprocessing) {
        SCOPED_TIMER("Partitioning");
        SCOPED_TIMER("Postprocessing");

        const NodeID num_isolated_nodes = graph::integrate_isolated_nodes(graph, original_epsilon, ctx);
        p_graph = graph::assign_isolated_nodes(std::move(p_graph), num_isolated_nodes, ctx.partition);
    }

    // Store output partition (if requested)
    if (ctx.save_partition) {
        SCOPED_TIMER("IO");
        shm::io::partition::write(ctx.partition_file(), p_graph, permutations.old_to_new);
        LOG << "Wrote partition to: " << ctx.partition_file();
    }

    // Print some statistics
    STOP_TIMER(); // stop root timer
    cio::print_banner("Statistics");
    print_statistics(p_graph, ctx);
    return 0;
}
