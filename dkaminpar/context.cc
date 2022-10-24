/*******************************************************************************
 * @file:   context.cc
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Context struct for the distributed graph partitioner.
 ******************************************************************************/
#include "dkaminpar/context.h"

#include <unordered_map>

#include <tbb/parallel_for.h>

#include "dkaminpar/mpi/wrapper.h"

namespace kaminpar::dist {
using namespace std::string_literals;

//
// Functions for string <-> enum conversion
//

std::unordered_map<std::string, PartitioningMode> get_partitioning_modes() {
    return {
        {"deep", PartitioningMode::DEEP},
        {"deeper", PartitioningMode::DEEPER},
        {"kway", PartitioningMode::KWAY},
    };
}

std::ostream& operator<<(std::ostream& out, const PartitioningMode mode) {
    switch (mode) {
        case PartitioningMode::DEEP:
            return out << "deep";
        case PartitioningMode::DEEPER:
            return out << "deeper";
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

//
// Functions for compact, parsable context output
//

void LabelPropagationCoarseningContext::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "num_iterations=" << num_iterations << " "                                             //
        << prefix << "active_high_degree_threshold=" << active_high_degree_threshold << " "                 //
        << prefix << "passive_high_degree_threshold=" << passive_high_degree_threshold << " "               //
        << prefix << "max_num_neighbors=" << max_num_neighbors << " "                                       //
        << prefix << "merge_singleton_clusters=" << merge_singleton_clusters << " "                         //
        << prefix << "merge_nonadjacent_clusters_threshold=" << merge_nonadjacent_clusters_threshold << " " //
        << prefix << "total_num_chunks=" << total_num_chunks << " "                                         //
        << prefix << "num_chunks=" << num_chunks << " "                                                     //
        << prefix << "min_num_chunks=" << min_num_chunks << " "                                             //
        << prefix << "ignore_ghost_nodes=" << ignore_ghost_nodes << " "                                     //
        << prefix << "keep_ghost_clusters=" << keep_ghost_clusters << " ";                                  //
}

void LabelPropagationRefinementContext::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "active_high_degree_threshold=" << active_high_degree_threshold << " " //
        << prefix << "num_iterations=" << num_iterations << " "                             //
        << prefix << "total_num_chunks=" << total_num_chunks << " "                         //
        << prefix << "num_chunks=" << num_chunks << " "                                     //
        << prefix << "min_num_chunks=" << min_num_chunks << " "                             //
        << prefix << "num_move_attempts=" << num_move_attempts << " "                       //
        << prefix << "ignore_probabilities=" << ignore_probabilities << " ";                //
}

void FMRefinementContext::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "alpha=" << alpha << " "                      //
        << prefix << "distance=" << radius << " "                  //
        << prefix << "hops=" << pe_radius << " "                   //
        << prefix << "overlap_regions=" << overlap_regions << " "  //
        << prefix << "num_iterations=" << num_iterations << " "    //
        << prefix << "sequential=" << sequential << " "            //
        << prefix << "premove_locally=" << premove_locally << " "  //
        << prefix << "bound_degree=" << bound_degree << " "        //
        << prefix << "contract_border=" << contract_border << " "; //
}

void CoarseningContext::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "max_global_clustering_levels=" << max_global_clustering_levels << " " //
        << prefix << "global_clustering_algorithm=" << global_clustering_algorithm << " "   //
        << prefix << "global_contraction_algorithm=" << global_contraction_algorithm << " " //
        << prefix << "max_local_clustering_levels=" << max_local_clustering_levels << " "   //
        << prefix << "local_clustering_algorithm=" << local_clustering_algorithm << " "     //
        << prefix << "contraction_limit=" << contraction_limit << " "                       //
        << prefix << "cluster_weight_limit=" << cluster_weight_limit << " "                 //
        << prefix << "cluster_weight_multiplier=" << cluster_weight_multiplier << " ";      //
    local_lp.print_compact(out, prefix + "local_lp.");
    global_lp.print_compact(out, prefix + "global_lp.");
}

void BalancingContext::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "algorithm=" << algorithm << " "                      //
        << prefix << "num_nodes_per_block=" << num_nodes_per_block << " "; //
}

void MtKaHyParContext::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "preset_filename=" << preset_filename << " "; //
}

void InitialPartitioningContext::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "algorithm=" << algorithm << " ";
    mtkahypar.print_compact(out, prefix + "mtkahypar.");
    // kaminpar.print(out, prefix + "kaminpar.");
}

void RefinementContext::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "algorithm=" << algorithm << " ";
    lp.print_compact(out, prefix + "lp.");
    fm.print_compact(out, prefix + "fm.");
    balancing.print_compact(out, prefix + "balancing.");
}

void ParallelContext::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "num_threads=" << num_threads << " "                                         //
        << prefix << "num_mpis=" << num_mpis << " "                                               //
        << prefix << "use_interleaved_numa_allocation=" << use_interleaved_numa_allocation << " " //
        << prefix << "mpi_thread_support=" << mpi_thread_support << " ";                          //
}

void PartitionContext::setup(const DistributedGraph& graph) {
    _global_n = graph.global_n();
    _global_m = graph.global_m();
    _global_total_node_weight =
        mpi::allreduce<GlobalNodeWeight>(graph.total_node_weight(), MPI_SUM, graph.communicator());
    _local_n                = graph.n();
    _total_n                = graph.total_n();
    _local_m                = graph.m();
    _total_node_weight      = graph.total_node_weight();
    _global_max_node_weight = graph.global_max_node_weight();

    setup_perfectly_balanced_block_weights();
    setup_max_block_weights();
}

void PartitionContext::setup(const shm::Graph& graph) {
    _global_n                 = graph.n();
    _global_m                 = graph.m();
    _global_total_node_weight = graph.total_node_weight();
    _local_n                  = graph.n();
    _total_n                  = graph.n();
    _local_m                  = graph.m();
    _total_node_weight        = graph.total_node_weight();
    _global_max_node_weight   = graph.max_node_weight();

    setup_perfectly_balanced_block_weights();
    setup_max_block_weights();
}

void PartitionContext::setup_perfectly_balanced_block_weights() {
    _perfectly_balanced_block_weights.resize(k);

    const BlockWeight perfectly_balanced_block_weight = std::ceil(static_cast<double>(global_total_node_weight()) / k);
    tbb::parallel_for<BlockID>(0, k, [&](const BlockID b) {
        _perfectly_balanced_block_weights[b] = perfectly_balanced_block_weight;
    });
}

void PartitionContext::setup_max_block_weights() {
    _max_block_weights.resize(k);

    tbb::parallel_for<BlockID>(0, k, [&](const BlockID b) {
        const BlockWeight max_eps_weight =
            static_cast<BlockWeight>((1.0 + epsilon) * static_cast<double>(perfectly_balanced_block_weight(b)));
        const BlockWeight max_abs_weight = perfectly_balanced_block_weight(b) + _global_max_node_weight;

        // Only relax weight on coarse levels
        if (static_cast<GlobalNodeWeight>(_global_n) == _global_total_node_weight) {
            _max_block_weights[b] = max_eps_weight;
        } else {
            _max_block_weights[b] = std::max(max_eps_weight, max_abs_weight);
        }
    });
}

void PartitionContext::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "k=" << k << " "             //
        << prefix << "k_prime=" << k_prime << " " //
        << prefix << "epsilon=" << epsilon << " " //
        << prefix << "mode=" << mode << " ";      //
}

void DebugContext::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "save_imbalanced_partitions=" << save_imbalanced_partitions << " " //
        << prefix << "save_graph_hierarchy=" << save_graph_hierarchy << " "             //
        << prefix << "save_coarsest_graph=" << save_coarsest_graph << " "               //
        << prefix << "save_clustering_hierarchy=" << save_clustering_hierarchy << " ";  //
}

void Context::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "graph_filename=" << graph_filename << " "         //
        << prefix << "load_edge_balanced=" << load_edge_balanced << " " //
        << prefix << "seed=" << seed << " "                             //
        << prefix << "quiet=" << quiet << " "                           //
        << prefix << "num_repetitions=" << num_repetitions << " "       //
        << prefix << "sort_graph=" << sort_graph << " "                 //
        << prefix << "time_limit=" << time_limit << " ";                //
    partition.print_compact(out, prefix + "partition.");
    parallel.print_compact(out, prefix + "parallel.");
    coarsening.print_compact(out, prefix + "coarsening.");
    initial_partitioning.print_compact(out, prefix + "initial_partitioning.");
    refinement.print_compact(out, prefix + "refinement.");
}
} // namespace kaminpar::dist
