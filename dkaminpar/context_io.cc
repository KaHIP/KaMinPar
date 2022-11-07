/*******************************************************************************
 * @file:   context_io.cc
 * @author: Daniel Seemaier
 * @date:   27.10.2022
 * @brief:  Utility functions to read/write parts of the partitioner context
 * from/to strings.
 ******************************************************************************/
#include "dkaminpar/context_io.h"

#include <iomanip>
#include <ostream>
#include <unordered_map>

#include "dkaminpar/context.h"

#include "common/console_io.h"

namespace kaminpar::dist {
std::unordered_map<std::string, PartitioningMode> get_partitioning_modes() {
    return {
        {"deep", PartitioningMode::DEEP},
        {"kway", PartitioningMode::KWAY},
    };
}

std::ostream& operator<<(std::ostream& out, const PartitioningMode mode) {
    switch (mode) {
        case PartitioningMode::DEEP:
            return out << "deep";
        case PartitioningMode::KWAY:
            return out << "kway";
    }

    return out << "<invalid>";
}

std::unordered_map<std::string, GlobalClusteringAlgorithm> get_global_clustering_algorithms() {
    return {
        {"noop", GlobalClusteringAlgorithm::NOOP},
        {"lp", GlobalClusteringAlgorithm::LP},
        {"active-set-lp", GlobalClusteringAlgorithm::ACTIVE_SET_LP},
        {"locking-lp", GlobalClusteringAlgorithm::LOCKING_LP},
    };
}

std::ostream& operator<<(std::ostream& out, const GlobalClusteringAlgorithm algorithm) {
    switch (algorithm) {
        case GlobalClusteringAlgorithm::NOOP:
            return out << "noop";
        case GlobalClusteringAlgorithm::LP:
            return out << "lp";
        case GlobalClusteringAlgorithm::ACTIVE_SET_LP:
            return out << "active-set-lp";
        case GlobalClusteringAlgorithm::LOCKING_LP:
            return out << "locking-lp";
    }

    return out << "<invalid>";
}

std::unordered_map<std::string, LocalClusteringAlgorithm> get_local_clustering_algorithms() {
    return {
        {"noop", LocalClusteringAlgorithm::NOOP},
        {"lp", LocalClusteringAlgorithm::LP},
    };
}

std::ostream& operator<<(std::ostream& out, const LocalClusteringAlgorithm algorithm) {
    switch (algorithm) {
        case LocalClusteringAlgorithm::NOOP:
            return out << "noop";
        case LocalClusteringAlgorithm::LP:
            return out << "lp";
    }

    return out << "<invalid>";
}

std::unordered_map<std::string, GlobalContractionAlgorithm> get_global_contraction_algorithms() {
    return {
        {"no-migration", GlobalContractionAlgorithm::NO_MIGRATION},
        {"minimal-migration", GlobalContractionAlgorithm::MINIMAL_MIGRATION},
        {"full-migration", GlobalContractionAlgorithm::FULL_MIGRATION},
    };
}

std::ostream& operator<<(std::ostream& out, const GlobalContractionAlgorithm algorithm) {
    switch (algorithm) {
        case GlobalContractionAlgorithm::NO_MIGRATION:
            return out << "no-migration";
        case GlobalContractionAlgorithm::MINIMAL_MIGRATION:
            return out << "minimal-migration";
        case GlobalContractionAlgorithm::FULL_MIGRATION:
            return out << "full-migration";
    }

    return out << "<invalid>";
}

std::unordered_map<std::string, InitialPartitioningAlgorithm> get_initial_partitioning_algorithms() {
    return {
        {"kaminpar", InitialPartitioningAlgorithm::KAMINPAR},
        {"mtkahypar", InitialPartitioningAlgorithm::MTKAHYPAR},
        {"random", InitialPartitioningAlgorithm::RANDOM},
    };
}

std::ostream& operator<<(std::ostream& out, const InitialPartitioningAlgorithm algorithm) {
    switch (algorithm) {
        case InitialPartitioningAlgorithm::KAMINPAR:
            return out << "kaminpar";
        case InitialPartitioningAlgorithm::MTKAHYPAR:
            return out << "mtkahypar";
        case InitialPartitioningAlgorithm::RANDOM:
            return out << "random";
    }

    return out << "<invalid>";
}

std::unordered_map<std::string, KWayRefinementAlgorithm> get_kway_refinement_algorithms() {
    return {
        {"noop", KWayRefinementAlgorithm::NOOP},
        {"lp", KWayRefinementAlgorithm::LP},
        {"local-fm", KWayRefinementAlgorithm::LOCAL_FM},
        {"fm", KWayRefinementAlgorithm::FM},
        {"lp+local-fm", KWayRefinementAlgorithm::LP_THEN_LOCAL_FM},
        {"lp+fm", KWayRefinementAlgorithm::LP_THEN_FM},
    };
}

std::ostream& operator<<(std::ostream& out, const KWayRefinementAlgorithm algorithm) {
    switch (algorithm) {
        case KWayRefinementAlgorithm::NOOP:
            return out << "noop";
        case KWayRefinementAlgorithm::LP:
            return out << "lp";
        case KWayRefinementAlgorithm::LOCAL_FM:
            return out << "local-fm";
        case KWayRefinementAlgorithm::FM:
            return out << "fm";
        case KWayRefinementAlgorithm::LP_THEN_LOCAL_FM:
            return out << "lp+local-fm";
        case KWayRefinementAlgorithm::LP_THEN_FM:
            return out << "lp+fm";
    }

    return out << "<invalid>";
}

std::unordered_map<std::string, BalancingAlgorithm> get_balancing_algorithms() {
    return {
        {"distributed", BalancingAlgorithm::DISTRIBUTED},
    };
}

std::ostream& operator<<(std::ostream& out, const BalancingAlgorithm algorithm) {
    switch (algorithm) {
        case BalancingAlgorithm::DISTRIBUTED:
            return out << "distributed";
    }

    return out << "<invalid>";
}

void print_compact(const LabelPropagationCoarseningContext& ctx, std::ostream& out, const std::string& prefix) {
    out << prefix << "num_iterations=" << ctx.num_iterations << " "                                             //
        << prefix << "active_high_degree_threshold=" << ctx.active_high_degree_threshold << " "                 //
        << prefix << "passive_high_degree_threshold=" << ctx.passive_high_degree_threshold << " "               //
        << prefix << "max_num_neighbors=" << ctx.max_num_neighbors << " "                                       //
        << prefix << "merge_singleton_clusters=" << ctx.merge_singleton_clusters << " "                         //
        << prefix << "merge_nonadjacent_clusters_threshold=" << ctx.merge_nonadjacent_clusters_threshold << " " //
        << prefix << "total_num_chunks=" << ctx.total_num_chunks << " "                                         //
        << prefix << "num_chunks=" << ctx.num_chunks << " "                                                     //
        << prefix << "min_num_chunks=" << ctx.min_num_chunks << " "                                             //
        << prefix << "ignore_ghost_nodes=" << ctx.ignore_ghost_nodes << " "                                     //
        << prefix << "keep_ghost_clusters=" << ctx.keep_ghost_clusters << " ";                                  //
}

void print_compact(const LabelPropagationRefinementContext& ctx, std::ostream& out, const std::string& prefix) {
    out << prefix << "active_high_degree_threshold=" << ctx.active_high_degree_threshold << " " //
        << prefix << "num_iterations=" << ctx.num_iterations << " "                             //
        << prefix << "total_num_chunks=" << ctx.total_num_chunks << " "                         //
        << prefix << "num_chunks=" << ctx.num_chunks << " "                                     //
        << prefix << "min_num_chunks=" << ctx.min_num_chunks << " "                             //
        << prefix << "num_move_attempts=" << ctx.num_move_attempts << " "                       //
        << prefix << "ignore_probabilities=" << ctx.ignore_probabilities << " ";                //
}

void print_compact(const FMRefinementContext& ctx, std::ostream& out, const std::string& prefix) {
    out << prefix << "alpha=" << ctx.alpha << " "                      //
        << prefix << "distance=" << ctx.radius << " "                  //
        << prefix << "hops=" << ctx.pe_radius << " "                   //
        << prefix << "overlap_regions=" << ctx.overlap_regions << " "  //
        << prefix << "num_iterations=" << ctx.num_iterations << " "    //
        << prefix << "sequential=" << ctx.sequential << " "            //
        << prefix << "premove_locally=" << ctx.premove_locally << " "  //
        << prefix << "bound_degree=" << ctx.bound_degree << " "        //
        << prefix << "contract_border=" << ctx.contract_border << " "; //
}

void print_compact(const CoarseningContext& ctx, std::ostream& out, const std::string& prefix) {
    out << prefix << "max_global_clustering_levels=" << ctx.max_global_clustering_levels << " " //
        << prefix << "global_clustering_algorithm=" << ctx.global_clustering_algorithm << " "   //
        << prefix << "global_contraction_algorithm=" << ctx.global_contraction_algorithm << " " //
        << prefix << "max_local_clustering_levels=" << ctx.max_local_clustering_levels << " "   //
        << prefix << "local_clustering_algorithm=" << ctx.local_clustering_algorithm << " "     //
        << prefix << "contraction_limit=" << ctx.contraction_limit << " "                       //
        << prefix << "cluster_weight_limit=" << ctx.cluster_weight_limit << " "                 //
        << prefix << "cluster_weight_multiplier=" << ctx.cluster_weight_multiplier << " ";      //
    print_compact(ctx.local_lp, out, prefix + "local_lp.");
    print_compact(ctx.global_lp, out, prefix + "global_lp.");
}

void print_compact(const BalancingContext& ctx, std::ostream& out, const std::string& prefix) {
    out << prefix << "algorithm=" << ctx.algorithm << " "                      //
        << prefix << "num_nodes_per_block=" << ctx.num_nodes_per_block << " "; //
}

void print_compact(const MtKaHyParContext& ctx, std::ostream& out, const std::string& prefix) {
    out << prefix << "preset_filename=" << ctx.preset_filename << " "; //
}

void print_compact(const InitialPartitioningContext& ctx, std::ostream& out, const std::string& prefix) {
    out << prefix << "algorithm=" << ctx.algorithm << " ";
    print_compact(ctx.mtkahypar, out, prefix + "mtkahypar.");

    // Currently disabled because it produces too much output:
    // kaminpar.print(out, prefix + "kaminpar.");
}

void print_compact(const RefinementContext& ctx, std::ostream& out, const std::string& prefix) {
    out << prefix << "algorithm=" << ctx.algorithm << " ";
    print_compact(ctx.lp, out, prefix + "lp.");
    print_compact(ctx.fm, out, prefix + "fm.");
    print_compact(ctx.balancing, out, prefix + "balancing.");
}

void print_compact(const ParallelContext& ctx, std::ostream& out, const std::string& prefix) {
    out << prefix << "num_threads=" << ctx.num_threads << " "                                          //
        << prefix << "num_mpis=" << ctx.num_mpis << " "                                                //
        << prefix << "use_interleaved_numa_allocation=" << ctx.use_interleaved_numa_allocation << " "; //
}

void print_compact(const PartitionContext& ctx, std::ostream& out, const std::string& prefix) {
    out << prefix << "k=" << ctx.k << " "                                          //
        << prefix << "K=" << ctx.K << " "                                          //
        << prefix << "epsilon=" << ctx.epsilon << " "                              //
        << prefix << "mode=" << ctx.mode << " "                                    //
        << prefix << "enable_pe_splitting=" << ctx.enable_pe_splitting << " "      //
        << prefix << "simulate_singlethread=" << ctx.simulate_singlethread << " "; //
}

void print_compact(const DebugContext& ctx, std::ostream& out, const std::string& prefix) {
    out << prefix << "save_finest_graph=" << ctx.save_finest_graph << " "                 //
        << prefix << "save_coarsest_graph=" << ctx.save_coarsest_graph << " "             //
        << prefix << "save_graph_hierarchy=" << ctx.save_graph_hierarchy << " "           //
        << prefix << "save_clustering_hierarchy=" << ctx.save_clustering_hierarchy << " " //
        << prefix << "save_partition_hierarchy=" << ctx.save_partition_hierarchy << " ";  //
}

void print_compact(const Context& ctx, std::ostream& out, const std::string& prefix) {
    out << prefix << "graph_filename=" << ctx.graph_filename << " "         //
        << prefix << "load_edge_balanced=" << ctx.load_edge_balanced << " " //
        << prefix << "seed=" << ctx.seed << " "                             //
        << prefix << "quiet=" << ctx.quiet << " "                           //
        << prefix << "num_repetitions=" << ctx.num_repetitions << " "       //
        << prefix << "sort_graph=" << ctx.sort_graph << " "                 //
        << prefix << "time_limit=" << ctx.time_limit << " ";                //
    print_compact(ctx.partition, out, prefix + "partition.");
    print_compact(ctx.parallel, out, prefix + "parallel.");
    print_compact(ctx.coarsening, out, prefix + "coarsening.");
    print_compact(ctx.initial_partitioning, out, prefix + "initial_partitioning.");
    print_compact(ctx.refinement, out, prefix + "refinement.");
}

void print(const Context& ctx, const bool root, std::ostream& out) {
    if (root) {
        out << "Seed:                         " << ctx.seed << "\n";
        if (!ctx.graph_filename.empty()) {
            out << "Graph:                        " << ctx.graph_filename << "\n";
        }
    }
    print(ctx.partition, root, out);
    if (root) {
        cio::print_delimiter(out, '-');
        print(ctx.coarsening, out);
        cio::print_delimiter(out, '-');
        print(ctx.initial_partitioning, out);
        cio::print_delimiter(out, '-');
        print(ctx.refinement, out);
    }
}

void print(const PartitionContext& ctx, const bool root, std::ostream& out) {
    // If the graph context has not been initialized with a graph, be silent
    // (This should never happen)
    if (!ctx.graph.initialized()) {
        return;
    }

    const auto size  = std::max<std::uint64_t>({
         static_cast<std::uint64_t>(ctx.graph.global_n()),
         static_cast<std::uint64_t>(ctx.graph.global_m()),
         static_cast<std::uint64_t>(ctx.graph.max_block_weight(0)),
    });
    const auto width = std::ceil(std::log10(size)) + 1;

    if (root) {
        out << "  Number of global nodes:    " << std::setw(width) << ctx.graph.global_n();
        if (asserting_cast<NodeWeight>(ctx.graph.global_n()) == ctx.graph.global_total_node_weight()) {
            out << " (unweighted)\n";
        } else {
            out << " (total weight: " << ctx.graph.global_total_node_weight() << ")\n";
        }
        out << "  Number of global edges:    " << std::setw(width) << ctx.graph.global_m();
        if (asserting_cast<EdgeWeight>(ctx.graph.global_m()) == ctx.graph.global_total_edge_weight()) {
            out << " (unweighted)\n";
        } else {
            out << " (total weight: " << ctx.graph.global_total_edge_weight() << ")\n";
        }
        out << "Number of blocks:             " << ctx.k << "\n";
        out << "Maximum block weight:         " << ctx.graph.max_block_weight(0) << " ("
            << ctx.graph.perfectly_balanced_block_weight(0) << " + " << 100 * ctx.epsilon << "%)\n";

        cio::print_delimiter(out, '-');

        out << "Partitioning mode:            " << ctx.mode << "\n";
        if (ctx.mode == PartitioningMode::DEEP) {
            out << "  Enable PE-splitting:        " << (ctx.enable_pe_splitting ? "yes" : "no") << "\n";
            out << "  Partition extension factor: " << ctx.K << "\n";
            out << "  Simulate seq. hybrid exe.:  " << (ctx.simulate_singlethread ? "yes" : "no") << "\n";
        }
    }
}

void print(const CoarseningContext& ctx, std::ostream& out) {
    if (ctx.max_global_clustering_levels > 0 && ctx.max_local_clustering_levels > 0) {
        out << "Coarsening mode:              local[" << ctx.max_local_clustering_levels << "]+global["
            << ctx.max_global_clustering_levels << "]\n";
    } else if (ctx.max_global_clustering_levels > 0) {
        out << "Coarsening mode:              global[" << ctx.max_global_clustering_levels << "]\n";
    } else if (ctx.max_local_clustering_levels > 0) {
        out << "Coarsening mode:              local[" << ctx.max_local_clustering_levels << "]\n";
    } else {
        out << "Coarsening mode:              disabled\n";
    }
    if (ctx.max_local_clustering_levels > 0) {
        out << "Local clustering algorithm:   " << ctx.local_clustering_algorithm << "\n";
        out << "  Number of iterations:       " << ctx.local_lp.num_iterations << "\n";
        out << "  High degree threshold:      " << ctx.local_lp.passive_high_degree_threshold << " (passive), "
            << ctx.local_lp.active_high_degree_threshold << " (active)\n";
        out << "  Max degree:                 " << ctx.local_lp.max_num_neighbors << "\n";
        out << "  Ghost nodes:                " << (ctx.local_lp.ignore_ghost_nodes ? "ignore" : "consider") << "+"
            << (ctx.local_lp.keep_ghost_clusters ? "keep" : "discard") << "\n";
    }
    if (ctx.max_global_clustering_levels > 0) {
        out << "Global clustering algorithm:  " << ctx.global_clustering_algorithm << "\n";
        out << "  Number of iterations:       " << ctx.global_lp.num_iterations << "\n";
        out << "  High degree threshold:      " << ctx.global_lp.passive_high_degree_threshold << " (passive), "
            << ctx.global_lp.active_high_degree_threshold << " (active)\n";
        out << "  Max degree:                 " << ctx.global_lp.max_num_neighbors << "\n";
        out << "  Number of chunks:           " << ctx.global_lp.num_chunks << " (min: " << ctx.global_lp.min_num_chunks
            << ", total: " << ctx.global_lp.total_num_chunks << ")"
            << (ctx.global_lp.scale_chunks_with_threads ? ", scaled" : "") << "\n";
    }
}

void print(const InitialPartitioningContext& ctx, std::ostream& out) {
    out << "IP algorithm:                 " << ctx.algorithm << "\n";
    if (ctx.algorithm == InitialPartitioningAlgorithm::KAMINPAR) {
        out << "  Configuration preset:       default\n";
    } else if (ctx.algorithm == InitialPartitioningAlgorithm::MTKAHYPAR) {
        out << "  Configuration file:         " << ctx.mtkahypar.preset_filename << "\n";
    }
}

void print(const RefinementContext& ctx, std::ostream& out) {
    out << "Refinement algorithm:         " << ctx.algorithm << "\n";
    out << "Refine initial partition:     " << (ctx.refine_coarsest_level ? "yes" : "no") << "\n";
    if (ctx.algorithm == KWayRefinementAlgorithm::LP || ctx.algorithm == KWayRefinementAlgorithm::LP_THEN_FM) {
        out << "Label propagation:\n";
        out << "  Number of iterations:       " << ctx.lp.num_iterations << "\n";
        out << "  Number of chunks:           " << ctx.lp.num_chunks << " (min: " << ctx.lp.min_num_chunks
            << ", total: " << ctx.lp.total_num_chunks << ")" << (ctx.lp.scale_chunks_with_threads ? ", scaled" : "")
            << "\n";
        out << "  Use probabilistic moves:    " << (ctx.lp.ignore_probabilities ? "no" : "yes") << "\n";
        out << "  Number of retries:          " << ctx.lp.num_move_attempts << "\n";
    }
    out << "Balancing algorithm:          " << ctx.balancing.algorithm << "\n";
    if (ctx.balancing.algorithm == BalancingAlgorithm::DISTRIBUTED) {
        out << "  Number of nodes per block:  " << ctx.balancing.num_nodes_per_block << "\n";
    }
}
} // namespace kaminpar::dist
