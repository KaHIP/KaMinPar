/*******************************************************************************
 * @file:   context.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Configuration struct for KaMinPar.
 ******************************************************************************/
#include "kaminpar/context.h"

#include <iomanip>
#include <unordered_map>

#include <kassert/kassert.hpp>

#include "common/asserting_cast.h"
#include "common/console_io.h"
#include "common/math.h"

namespace kaminpar::shm {
using namespace std::string_literals;

//
// std::string <-> enum conversion
//

std::unordered_map<std::string, ClusteringAlgorithm> get_clustering_algorithms() {
    return {
        {"noop", ClusteringAlgorithm::NOOP},
        {"lp", ClusteringAlgorithm::LABEL_PROPAGATION},
    };
}

std::ostream& operator<<(std::ostream& out, const ClusteringAlgorithm algorithm) {
    switch (algorithm) {
        case ClusteringAlgorithm::NOOP:
            return out << "noop";
        case ClusteringAlgorithm::LABEL_PROPAGATION:
            return out << "lp";
    }
    return out << "<invalid>";
}

std::unordered_map<std::string, ClusterWeightLimit> get_cluster_weight_limits() {
    return {
        {"epsilon-block-weight", ClusterWeightLimit::EPSILON_BLOCK_WEIGHT},
        {"static-block-weight", ClusterWeightLimit::BLOCK_WEIGHT},
        {"one", ClusterWeightLimit::ONE},
        {"zero", ClusterWeightLimit::ZERO},
    };
}

std::ostream& operator<<(std::ostream& out, const ClusterWeightLimit limit) {
    switch (limit) {
        case ClusterWeightLimit::EPSILON_BLOCK_WEIGHT:
            return out << "epsilon-block-weight";
        case ClusterWeightLimit::BLOCK_WEIGHT:
            return out << "static-block-weight";
        case ClusterWeightLimit::ONE:
            return out << "one";
        case ClusterWeightLimit::ZERO:
            return out << "zero";
    }
    return out << "<invalid>";
}

std::unordered_map<std::string, RefinementAlgorithm> get_2way_refinement_algorithms() {
    return {
        {"noop", RefinementAlgorithm::NOOP},
        {"fm", RefinementAlgorithm::TWO_WAY_FM},
    };
}

std::unordered_map<std::string, RefinementAlgorithm> get_kway_refinement_algorithms() {
    return {
        {"noop", RefinementAlgorithm::NOOP},
        {"lp", RefinementAlgorithm::LABEL_PROPAGATION},
    };
}

std::ostream& operator<<(std::ostream& out, const RefinementAlgorithm algorithm) {
    switch (algorithm) {
        case RefinementAlgorithm::NOOP:
            return out << "noop";
        case RefinementAlgorithm::TWO_WAY_FM:
            return out << "fm";
        case RefinementAlgorithm::LABEL_PROPAGATION:
            return out << "lp";
    }

    return out << "<invalid>";
}

std::unordered_map<std::string, FMStoppingRule> get_fm_stopping_rules() {
    return {
        {"simple", FMStoppingRule::SIMPLE},
        {"adaptive", FMStoppingRule::ADAPTIVE},
    };
}

std::ostream& operator<<(std::ostream& out, const FMStoppingRule rule) {
    switch (rule) {
        case FMStoppingRule::SIMPLE:
            return out << "simple";
        case FMStoppingRule::ADAPTIVE:
            return out << "adaptive";
    }

    return out << "<invalid>";
}

std::unordered_map<std::string, BalancingTimepoint> get_balancing_timepoints() {
    return {
        {"before-refinement", BalancingTimepoint::BEFORE_KWAY_REFINEMENT},
        {"after-refinement", BalancingTimepoint::AFTER_KWAY_REFINEMENT},
        {"always", BalancingTimepoint::ALWAYS},
        {"never", BalancingTimepoint::NEVER},
    };
}

std::ostream& operator<<(std::ostream& out, const BalancingTimepoint timepoint) {
    switch (timepoint) {
        case BalancingTimepoint::BEFORE_KWAY_REFINEMENT:
            return out << "before-refinement";
        case BalancingTimepoint::AFTER_KWAY_REFINEMENT:
            return out << "after-refinement";
        case BalancingTimepoint::ALWAYS:
            return out << "aways";
        case BalancingTimepoint::NEVER:
            return out << "never";
    }

    return out << "<invalid>";
}

std::unordered_map<std::string, BalancingAlgorithm> get_balancing_algorithms() {
    return {
        {"noop", BalancingAlgorithm::NOOP},
        {"greedy", BalancingAlgorithm::BLOCK_LEVEL_PARALLEL_BALANCER},
    };
}

std::ostream& operator<<(std::ostream& out, const BalancingAlgorithm algorithm) {
    switch (algorithm) {
        case BalancingAlgorithm::NOOP:
            return out << "noop";
        case BalancingAlgorithm::BLOCK_LEVEL_PARALLEL_BALANCER:
            return out << "greedy";
    }

    return out << "<invalid>";
}

std::unordered_map<std::string, PartitioningMode> get_partitioning_modes() {
    return {
        {"deep", PartitioningMode::DEEP},
        {"rb", PartitioningMode::RB},
    };
}

std::ostream& operator<<(std::ostream& out, const PartitioningMode mode) {
    switch (mode) {
        case PartitioningMode::DEEP:
            return out << "deep";
        case PartitioningMode::RB:
            return out << "rb";
    }

    return out << "<invalid>";
}

std::unordered_map<std::string, InitialPartitioningMode> get_initial_partitioning_modes() {
    return {
        {"sequential", InitialPartitioningMode::SEQUENTIAL},
        {"async-parallel", InitialPartitioningMode::ASYNCHRONOUS_PARALLEL},
        {"sync-parallel", InitialPartitioningMode::SYNCHRONOUS_PARALLEL},
    };
}

std::ostream& operator<<(std::ostream& out, const InitialPartitioningMode mode) {
    switch (mode) {
        case InitialPartitioningMode::SEQUENTIAL:
            return out << "sequential";
        case InitialPartitioningMode::ASYNCHRONOUS_PARALLEL:
            return out << "async-parallel";
        case InitialPartitioningMode::SYNCHRONOUS_PARALLEL:
            return out << "sync-parallel";
    }

    return out << "<invalid>";
}

//
// PartitionContext
//

void PartitionContext::setup(const Graph& graph) {
    n                 = graph.n();
    m                 = graph.m();
    total_node_weight = graph.total_node_weight();
    total_edge_weight = graph.total_edge_weight();
    max_node_weight   = graph.max_node_weight();
    setup_block_weights();
}

void PartitionContext::setup_block_weights() {
    block_weights.setup(*this);
}

//
// BlockWeightsContext
//

void BlockWeightsContext::setup(const PartitionContext& p_ctx) {
    KASSERT(p_ctx.k != kInvalidBlockID, "PartitionContext::k not initialized");
    KASSERT(p_ctx.total_node_weight != kInvalidNodeWeight, "PartitionContext::total_node_weight not initialized");
    KASSERT(p_ctx.max_node_weight != kInvalidNodeWeight, "PartitionContext::max_node_weight not initialized");

    const auto perfectly_balanced_block_weight =
        static_cast<NodeWeight>(std::ceil(1.0 * p_ctx.total_node_weight / p_ctx.k));
    const auto max_block_weight = static_cast<NodeWeight>((1.0 + p_ctx.epsilon) * perfectly_balanced_block_weight);

    _max_block_weights.resize(p_ctx.k);
    _perfectly_balanced_block_weights.resize(p_ctx.k);

    tbb::parallel_for<BlockID>(0, p_ctx.k, [&](const BlockID b) {
        _perfectly_balanced_block_weights[b] = perfectly_balanced_block_weight;

        // relax balance constraint by max_node_weight on coarse levels only
        if (p_ctx.max_node_weight == 1) {
            _max_block_weights[b] = max_block_weight;
        } else {
            _max_block_weights[b] =
                std::max<NodeWeight>(max_block_weight, perfectly_balanced_block_weight + p_ctx.max_node_weight);
        }
    });
}

void BlockWeightsContext::setup(const PartitionContext& p_ctx, const scalable_vector<BlockID>& final_ks) {
    KASSERT(p_ctx.k != kInvalidBlockID, "PartitionContext::k not initialized");
    KASSERT(p_ctx.total_node_weight != kInvalidNodeWeight, "PartitionContext::total_node_weight not initialized");
    KASSERT(p_ctx.max_node_weight != kInvalidNodeWeight, "PartitionContext::max_node_weight not initialized");
    KASSERT(p_ctx.k == final_ks.size(), "bad number of blocks: got " << final_ks.size() << ", expected " << p_ctx.k);

    const BlockID final_k      = std::accumulate(final_ks.begin(), final_ks.end(), static_cast<BlockID>(0));
    const double  block_weight = 1.0 * p_ctx.total_node_weight / final_k;

    _max_block_weights.resize(p_ctx.k);
    _perfectly_balanced_block_weights.resize(p_ctx.k);

    tbb::parallel_for<BlockID>(0, final_ks.size(), [&](const BlockID b) {
        _perfectly_balanced_block_weights[b] = std::ceil(final_ks[b] * block_weight);

        const auto max_block_weight =
            static_cast<BlockWeight>((1.0 + p_ctx.epsilon) * _perfectly_balanced_block_weights[b]);

        // relax balance constraint by max_node_weight on coarse levels only
        if (p_ctx.max_node_weight == 1) {
            _max_block_weights[b] = max_block_weight;
        } else {
            _max_block_weights[b] =
                std::max<BlockWeight>(max_block_weight, _perfectly_balanced_block_weights[b] + p_ctx.max_node_weight);
        }
    });
}

[[nodiscard]] BlockWeight BlockWeightsContext::max(const BlockID b) const {
    KASSERT(b < _max_block_weights.size());
    return _max_block_weights[b];
}

[[nodiscard]] const scalable_vector<BlockWeight>& BlockWeightsContext::all_max() const {
    return _max_block_weights;
}

[[nodiscard]] BlockWeight BlockWeightsContext::perfectly_balanced(const BlockID b) const {
    KASSERT(b < _perfectly_balanced_block_weights.size());
    return _perfectly_balanced_block_weights[b];
}

[[nodiscard]] const scalable_vector<BlockWeight>& BlockWeightsContext::all_perfectly_balanced() const {
    return _perfectly_balanced_block_weights;
}

void Context::setup(const Graph& graph) {
    partition.setup(graph);
}

PartitionContext create_bipartition_context(
    const PartitionContext& k_p_ctx, const Graph& subgraph, const BlockID final_k1, const BlockID final_k2
) {
    PartitionContext two_p_ctx{};
    two_p_ctx.setup(subgraph);
    two_p_ctx.k       = 2;
    two_p_ctx.epsilon = compute_2way_adaptive_epsilon(k_p_ctx, subgraph.total_node_weight(), final_k1 + final_k2);
    two_p_ctx.block_weights.setup(two_p_ctx, {final_k1, final_k2});
    return two_p_ctx;
}

double compute_2way_adaptive_epsilon(
    const PartitionContext& p_ctx, const NodeWeight subgraph_total_node_weight, const BlockID subgraph_final_k
) {
    KASSERT(subgraph_final_k > 1u);

    const double base =
        (1.0 + p_ctx.epsilon) * subgraph_final_k * p_ctx.total_node_weight / p_ctx.k / subgraph_total_node_weight;
    const double exponent         = 1.0 / math::ceil_log2(subgraph_final_k);
    const double epsilon_prime    = std::pow(base, exponent) - 1.0;
    const double adaptive_epsilon = std::max(epsilon_prime, 0.0001);
    return adaptive_epsilon;
}

//
// Functions to print all parameters in a compact and parsable format
//

void PartitionContext::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "mode=" << mode << " "       //
        << prefix << "epsilon=" << epsilon << " " //
        << prefix << "k=" << k << " ";            //
}

void CoarseningContext::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "algorithm=" << algorithm << " "                                  //
        << prefix << "contraction_limit=" << contraction_limit << " "                  //
        << prefix << "enforce_contraction_limit=" << enforce_contraction_limit << " "  //
        << prefix << "convergence_threshold=" << convergence_threshold << " "          //
        << prefix << "cluster_weight_limit=" << cluster_weight_limit << " "            //
        << prefix << "cluster_weight_multiplier=" << cluster_weight_multiplier << " "; //
    lp.print_compact(out, prefix + "lp.");
}

void LabelPropagationCoarseningContext::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "num_iterations=" << num_iterations << " "                             //
        << prefix << "max_degree=" << large_degree_threshold << " "                         //
        << prefix << "two_hop_clustering_threshold=" << two_hop_clustering_threshold << " " //
        << prefix << "max_num_neighbors=" << max_num_neighbors << " ";                      //
}

void LabelPropagationRefinementContext::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "num_iterations=" << num_iterations << " "        //
        << prefix << "max_degree=" << large_degree_threshold << " "    //
        << prefix << "max_num_neighbors=" << max_num_neighbors << " "; //
}

void FMRefinementContext::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "stopping_rule=" << stopping_rule << " "             //
        << prefix << "num_fruitless_moves=" << num_fruitless_moves << " " //
        << prefix << "alpha=" << alpha << " ";                            //
}

void BalancerRefinementContext::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "timepoint=" << timepoint << " "  //
        << prefix << "algorithm=" << algorithm << " "; //
}

void RefinementContext::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "algorithm=" << algorithm << " "; //

    lp.print_compact(out, prefix + "lp.");
    fm.print_compact(out, prefix + "fm.");
    balancer.print_compact(out, prefix + "balancer.");
}

void InitialPartitioningContext::print_compact(std::ostream& out, const std::string& prefix) const {
    coarsening.print_compact(out, prefix + "coarsening.");
    refinement.print_compact(out, prefix + "refinement.");
    out << prefix << "mode=" << mode << " "                                                                 //
        << prefix << "repetition_multiplier=" << repetition_multiplier << " "                               //
        << prefix << "min_num_repetitions=" << min_num_repetitions << " "                                   //
        << prefix << "max_num_repetitions=" << max_num_repetitions << " "                                   //
        << prefix << "num_seed_iterations=" << num_seed_iterations << " "                                   //
        << prefix << "use_adaptive_bipartitioner_selection=" << use_adaptive_bipartitioner_selection << " " //
        << prefix << "multiplier_exponent=" << multiplier_exponent << " ";                                  //
}

void DebugContext::print_compact(std::ostream& out, const std::string& prefix) const {
    ((void)out);
    ((void)prefix);
}

void ParallelContext::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "use_interleaved_numa_allocation=" << use_interleaved_numa_allocation << " " //
        << prefix << "num_threads=" << num_threads << " ";                                        //
}

void Context::print_compact(std::ostream& out, const std::string& prefix) const {
    out << prefix << "graph_filename=" << graph_filename << " "           //
        << prefix << "seed=" << seed << " "                               //
        << prefix << "save_output_partition=" << save_partition << " "    //
        << prefix << "partition_filename=" << partition_filename << " "   //
        << prefix << "partition_directory=" << partition_directory << " " //
        << prefix << "quiet=" << quiet << " ";                            //
                                                                          //
    partition.print_compact(out, prefix + "partition.");
    coarsening.print_compact(out, prefix + "coarsening.");
    initial_partitioning.print_compact(out, prefix + "initial_partitioning.");
    refinement.print_compact(out, prefix + "refinement.");
    debug.print_compact(out, prefix + "debug.");
    parallel.print_compact(out, prefix + "parallel.");
}

//
// Functions to print important parameters in a readable format
//

void PartitionContext::print(std::ostream& out) const {
    const BlockWeight  max_block_weight = block_weights.max(0);
    const std::int64_t size             = std::max<std::int64_t>({n, m, max_block_weight});
    const std::size_t  width            = std::ceil(std::log10(size));

    out << "  Number of nodes:            " << std::setw(width) << n;
    if (asserting_cast<NodeWeight>(n) == total_node_weight) {
        out << " (unweighted)\n";
    } else {
        out << " (total weight: " << total_node_weight << ")\n";
    }
    out << "  Number of edges:            " << std::setw(width) << m;
    if (asserting_cast<EdgeWeight>(m) == total_edge_weight) {
        out << " (unweighted)\n";
    } else {
        out << " (total weight: " << total_edge_weight << ")\n";
    }
    out << "Number of blocks:             " << k << "\n";
    out << "Maximum block weight:         " << block_weights.max(0) << " (" << block_weights.perfectly_balanced(0)
        << " + " << 100 * epsilon << "%)\n";
}

void Context::print(std::ostream& out) const {
    out << "Seed:                         " << seed << "\n";
    out << "Graph:                        " << graph_filename << "\n";
    partition.print(out);
    cio::print_delimiter();
    // @todo
}
} // namespace kaminpar::shm
