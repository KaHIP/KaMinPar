/*******************************************************************************
 * @file:   arguments.cc
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:
 ******************************************************************************/
#include "apps/dkaminpar/arguments.h"

#include <string>

#include "kaminpar/application/arguments.h"

namespace kaminpar::dist {
using namespace std::string_literals;

#ifdef KAMINPAR_ENABLE_GRAPHGEN
void create_graphgen_options(
    GeneratorContext& g_ctx, kaminpar::Arguments& args, const std::string& name, const std::string& prefix
) {
    args.group(name, prefix)
        .argument(
            prefix, "Graph generator, possible values: {" + generator_type_names() + "}.", &g_ctx.type,
            generator_type_from_string
        )
        .argument(prefix + "-n", "Number of nodes in the graph.", &g_ctx.n)
        .argument(prefix + "-m", "Number of edges in the graph.", &g_ctx.m)
        .argument(prefix + "-p", "Edge probability.", &g_ctx.p)
        .argument(prefix + "-gamma", "Power law exponent (depending on model)", &g_ctx.gamma)
        .argument(prefix + "-save-graph", "Write the generated graph to the hard disk.", &g_ctx.save_graph)
        .argument(prefix + "-scale", "Scaling factor for the generated graph (e.g., number of PEs).", &g_ctx.scale)
        .argument(
            prefix + "-periodic", "Use periodic boundary condition when generating RDG2D graphs.", &g_ctx.periodic
        )
        .argument(
            prefix + "-validate", "Validate the graph format before using it. Useful for debugging.",
            &g_ctx.validate_graph
        )
        .argument(
                prefix + "-stats", "Enable more statistics.", &g_ctx.advanced_stats
        )
        .argument(prefix + "-a", "R-MAT", &g_ctx.prob_a)
        .argument(prefix + "-b", "R-MAT", &g_ctx.prob_b)
        .argument(prefix + "-c", "R-MAT", &g_ctx.prob_c);
}
#endif

void create_coarsening_label_propagation_options(
    LabelPropagationCoarseningContext& lp_ctx, kaminpar::Arguments& args, const std::string& name,
    const std::string& prefix
) {
    args.group(name, prefix)
        .argument(prefix + "-iterations", "Maximum number of LP iterations.", &lp_ctx.num_iterations)
        .argument(
            prefix + "-total-num-chunks", "Number of communication chunks times number of PEs.",
            &lp_ctx.total_num_chunks
        )
        .argument(prefix + "-min-num-chunks", "Minimum number of communication chunks.", &lp_ctx.min_num_chunks)
        .argument(
            prefix + "-num-chunks",
            "Number of communication chunks. If set to 0, the value is computed from total-num-chunks.",
            &lp_ctx.num_chunks
        )
        .argument(
            prefix + "-ignore-ghost-nodes", "[Local] Ignore ghost nodes for cluster ratings", &lp_ctx.ignore_ghost_nodes
        )
        .argument(
            prefix + "-keep-ghost-clusters",
            "[Local] Instead of completely dissolving ghost clusters, remap them to a local cluster ID.",
            &lp_ctx.keep_ghost_clusters
        )
        .argument(
            prefix + "-active-large-degree-threshold",
            "[Global] Nodes with a degree larger than this are not moved to a new cluster.",
            &lp_ctx.active_high_degree_threshold
        )
        .argument(
            prefix + "-passive-large-degree-threshold",
            "[Global] Nodes with a degree larger than this are not considered when moving neighbors to a new cluster.",
            &lp_ctx.passive_high_degree_threshold
        );
}

void create_coarsening_options(
    CoarseningContext& c_ctx, kaminpar::Arguments& args, const std::string& name, const std::string& prefix
) {
    args.group(name, prefix)
        .argument(prefix + "-contraction-limit", "Contraction limit", &c_ctx.contraction_limit)
        .argument(
            prefix + "-max-local-levels", "Maximum number of local clustering levels.",
            &c_ctx.max_local_clustering_levels
        )
        .argument(
            prefix + "-max-global-levels", "Maximum number of global clustering levels.",
            &c_ctx.max_global_clustering_levels
        )
        .argument(
            prefix + "-global-clustering-algorithm",
            "Clustering algorithm, possible values: {"s + global_clustering_algorithm_names() + "}.",
            &c_ctx.global_clustering_algorithm, global_clustering_algorithm_from_string
        )
        .argument(
            prefix + "-global-contraction-algorithm",
            "Contraction algorithm, possible values: {"s + global_contraction_algorithm_names() + "}.",
            &c_ctx.global_contraction_algorithm, global_contraction_algorithm_from_string
        )
        .argument(
            prefix + "-local-clustering-algorithm",
            "Local clustering algorithm, possible values: {"s + local_clustering_algorithm_names() + "}.",
            &c_ctx.local_clustering_algorithm, local_clustering_algorithm_from_string
        )
        .argument(
            prefix + "-cluster-weight-limit",
            "Function to compute the cluster weight limit, possible values: {"s + shm::cluster_weight_limit_names()
                + "}.",
            &c_ctx.cluster_weight_limit, shm::cluster_weight_limit_from_string
        )
        .argument(
            prefix + "-cluster-weight-multiplier", "Multiplier for the cluster weight limit.",
            &c_ctx.cluster_weight_multiplier
        );

    create_coarsening_label_propagation_options(
        c_ctx.local_lp, args, name + " -> Local Label Propagation", prefix + "-llp"
    );
    create_coarsening_label_propagation_options(
        c_ctx.global_lp, args, name + " -> Global Label Propagation", prefix + "-glp"
    );
}

void create_balancing_options(
    BalancingContext& b_ctx, kaminpar::Arguments& args, const std::string& name, const std::string& prefix
) {
    args.group(name, prefix)
        .argument(prefix + "-algorithm", "Balancing algorithm", &b_ctx.algorithm, balancing_algorithm_from_string)
        .argument(
            prefix + "-num-nodes-per-block", "Number of nodes per block to keep in each reduction step",
            &b_ctx.num_nodes_per_block
        );
}

void create_refinement_label_propagation_options(
    LabelPropagationRefinementContext& lp_ctx, kaminpar::Arguments& args, const std::string& name,
    const std::string& prefix
) {
    args.group(name, prefix)
        .argument(prefix + "-iterations", "Maximum number of LP iterations.", &lp_ctx.num_iterations)
        .argument(
            prefix + "-total-num-chunks", "Number of communication chunks times number of PEs.",
            &lp_ctx.total_num_chunks
        )
        .argument(prefix + "-min-num-chunks", "Minimum number of communication chunks.", &lp_ctx.min_num_chunks)
        .argument(
            prefix + "-num-chunks",
            "Number of communication chunks. If set to 0, the value is computed from total-num-chunks.",
            &lp_ctx.num_chunks
        )
        .argument(
            prefix + "-active-large-degree-threshold",
            "Nodes with a degree larger than this are not moved to new blocks.", &lp_ctx.active_high_degree_threshold
        );
}

void create_refinement_fm_options(
    FMRefinementContext& fm_ctx, kaminpar::Arguments& args, const std::string& name, const std::string& prefix
) {
    args.group(name, prefix)
        .argument(prefix + "-alpha", "Alpha parameter for the adaptive stopping policy.", &fm_ctx.alpha)
        .argument(prefix + "-radius", "Search radius.", &fm_ctx.radius)
        .argument(prefix + "-pe-radius", "Search radius in number of PEs.", &fm_ctx.pe_radius)
        .argument(prefix + "-overlap-regions", "Overlap search regions.", &fm_ctx.overlap_regions)
        .argument(prefix + "-iterations", "Number of iterations to perform.", &fm_ctx.num_iterations)
        .argument(prefix + "-sequential", "Refine search regions sequentially.", &fm_ctx.sequential)
        .argument(
            prefix + "-premove-locally", "Move nodes right away, i.e., before global synchronization steps.",
            &fm_ctx.premove_locally
        )
        .argument(
            prefix + "-bound-degree", "Add at most this many neighbors of a high-degree node to a search region.",
            &fm_ctx.bound_degree
        )
        .argument(prefix + "-contract-border", "Contract border of search graphs", &fm_ctx.contract_border);
}

void create_refinement_options(
    RefinementContext& r_ctx, kaminpar::Arguments& args, const std::string& name, const std::string& prefix
) {
    args.group(name, prefix)
        .argument(
            prefix + "-algorithm",
            "Refinement algorithm, possible values: {"s + kway_refinement_algorithm_names() + "}.", &r_ctx.algorithm,
            kway_refinement_algorithm_from_string
        )
        .argument(prefix + "-coarsest", "Refine coarsest level", &r_ctx.refine_coarsest_level);
    create_refinement_label_propagation_options(r_ctx.lp, args, name + " -> Label Propagation", prefix + "-lp");
    create_refinement_fm_options(r_ctx.fm, args, name + " -> FM", prefix + "-fm");
    create_balancing_options(r_ctx.balancing, args, name + " -> Balancing", prefix + "-b");
}

void create_initial_partitioning_options(
    InitialPartitioningContext& i_ctx, kaminpar::Arguments& args, const std::string& name, const std::string& prefix
) {
    args.group(name, prefix)
        .argument(
            prefix + "-algorithm",
            "Initial partitioning algorithm, possible values: {"s + initial_partitioning_algorithm_names() + "}.",
            &i_ctx.algorithm, initial_partitioning_algorithm_from_string
        );
    shm::app::create_algorithm_options(i_ctx.sequential, args, "Initial Partitioning -> KaMinPar -> ", prefix + "i-");
}

void create_miscellaneous_context_options(
    Context& ctx, kaminpar::Arguments& args, const std::string& name, const std::string& prefix
) {
    args.group(name, prefix)
        .argument("epsilon", "Maximum allowed imbalance.", &ctx.partition.epsilon, 'e')
        .argument("threads", "Maximum number of threads to be used.", &ctx.parallel.num_threads, 't')
        .argument("seed", "Seed for random number generator.", &ctx.seed, 's')
        .argument("quiet", "Do not produce any output to stdout.", &ctx.quiet, 'q')
        .argument(
            "edge-balanced", "Read input graph such that edges are distributed evenly across PEs.",
            &ctx.load_edge_balanced, 'E'
        )
        .argument("repetitions", "Number of repetitions to perform.", &ctx.num_repetitions, 'R')
        .argument(
            "time-limit", "Time limit in seconds. Repeats partitioning until the time limit is exceeded.",
            &ctx.time_limit, 'T'
        )
        .argument("sort-graph", "Sort and rearrange the graph by degree buckets.", &ctx.sort_graph);
}

void create_mandatory_options(Context& ctx, kaminpar::Arguments& args, const std::string& name) {
    args.group(name, "", true)
        .argument("k", "Number of blocks", &ctx.partition.k, 'k')
        .argument("graph", "Graph to partition", &ctx.graph_filename, 'G');
}

void create_debug_options(
    DebugContext& d_ctx, kaminpar::Arguments& args, const std::string& name, const std::string& prefix
) {
    args.group(name, prefix)
        .argument(prefix + "-save-imbalanced-partitions", "", &d_ctx.save_imbalanced_partitions)
        .argument(prefix + "-save-graph-hierarchy", "", &d_ctx.save_graph_hierarchy)
        .argument(prefix + "-save-coarsest-graph", "", &d_ctx.save_coarsest_graph)
        .argument(prefix + "-save-clustering-hierarchy", "", &d_ctx.save_clustering_hierarchy);
}

void create_context_options(ApplicationContext& a_ctx, kaminpar::Arguments& args) {
    create_mandatory_options(a_ctx.ctx, args, "Mandatory");
    create_miscellaneous_context_options(a_ctx.ctx, args, "Miscellaneous", "m");
    create_coarsening_options(a_ctx.ctx.coarsening, args, "Coarsening", "c");
    create_initial_partitioning_options(a_ctx.ctx.initial_partitioning, args, "Initial Partitioning", "i");
    create_refinement_options(a_ctx.ctx.refinement, args, "Refinement", "r");
#ifdef KAMINPAR_ENABLE_GRAPHGEN
    create_graphgen_options(a_ctx.generator, args, "Graph Generation", "g");
#endif
    create_debug_options(a_ctx.ctx.debug, args, "Debug", "d");
}

ApplicationContext parse_options(int argc, char* argv[]) {
#ifdef KAMINPAR_ENABLE_GRAPHGEN
    ApplicationContext a_ctx{create_default_context(), {}};
#else
    ApplicationContext a_ctx{create_default_context()};
#endif
    kaminpar::Arguments arguments;
    create_context_options(a_ctx, arguments);
    arguments.parse(argc, argv);
    return a_ctx;
}
} // namespace kaminpar::dist
