/*******************************************************************************
 * @file:   context.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Configuration struct for KaMinPar.
 ******************************************************************************/
#include "kaminpar/context.h"

#include <kassert/kassert.hpp>

#include "common/utils/math.h"

namespace kaminpar::shm {
using namespace std::string_literals;

//
// Define std::string <-> enum conversion functions
//

DEFINE_ENUM_STRING_CONVERSION(ClusteringAlgorithm, clustering_algorithm) = {
    {ClusteringAlgorithm::NOOP, "noop"},
    {ClusteringAlgorithm::LABEL_PROPAGATION, "lp"},
};

DEFINE_ENUM_STRING_CONVERSION(ClusterWeightLimit, cluster_weight_limit) = {
    {ClusterWeightLimit::EPSILON_BLOCK_WEIGHT, "epsilon-block-weight"},
    {ClusterWeightLimit::BLOCK_WEIGHT, "static-block-weight"},
    {ClusterWeightLimit::ONE, "one"},
    {ClusterWeightLimit::ZERO, "zero"},
};

DEFINE_ENUM_STRING_CONVERSION(RefinementAlgorithm, refinement_algorithm) = {
    {RefinementAlgorithm::TWO_WAY_FM, "2way-fm"},
    {RefinementAlgorithm::LABEL_PROPAGATION, "lp"},
    {RefinementAlgorithm::NOOP, "noop"},
};

DEFINE_ENUM_STRING_CONVERSION(FMStoppingRule, fm_stopping_rule) = {
    {FMStoppingRule::SIMPLE, "simple"},
    {FMStoppingRule::ADAPTIVE, "adaptive"},
};

DEFINE_ENUM_STRING_CONVERSION(BalancingTimepoint, balancing_timepoint) = {
    {BalancingTimepoint::BEFORE_KWAY_REFINEMENT, "before-kway-ref"},
    {BalancingTimepoint::AFTER_KWAY_REFINEMENT, "after-kway-ref"},
    {BalancingTimepoint::ALWAYS, "always"},
    {BalancingTimepoint::NEVER, "never"},
};

DEFINE_ENUM_STRING_CONVERSION(BalancingAlgorithm, balancing_algorithm) = {
    {BalancingAlgorithm::BLOCK_LEVEL_PARALLEL_BALANCER, "block-parallel-balancer"},
};

DEFINE_ENUM_STRING_CONVERSION(PartitioningMode, partitioning_mode) = {
    {PartitioningMode::DEEP, "deep"},
    {PartitioningMode::RB, "rb"},
};

DEFINE_ENUM_STRING_CONVERSION(InitialPartitioningMode, initial_partitioning_mode) = {
    {InitialPartitioningMode::SEQUENTIAL, "sequential"},
    {InitialPartitioningMode::ASYNCHRONOUS_PARALLEL, "async-parallel"},
    {InitialPartitioningMode::SYNCHRONOUS_PARALLEL, "sync-parallel"},
};

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

//
// print() member functions
//

void PartitionContext::print(std::ostream& out, const std::string& prefix) const {
    out << prefix << "mode=" << mode << " "                                            //
        << prefix << "epsilon=" << epsilon << " "                                      //
        << prefix << "k=" << k << " "                                                  //
        << prefix << "fast_initial_partitioning=" << fast_initial_partitioning << " "; //
}

void CoarseningContext::print(std::ostream& out, const std::string& prefix) const {
    out << prefix << "algorithm=" << algorithm << " "                                  //
        << prefix << "contraction_limit=" << contraction_limit << " "                  //
        << prefix << "enforce_contraction_limit=" << enforce_contraction_limit << " "  //
        << prefix << "convergence_threshold=" << convergence_threshold << " "          //
        << prefix << "cluster_weight_limit=" << cluster_weight_limit << " "            //
        << prefix << "cluster_weight_multiplier=" << cluster_weight_multiplier << " "; //
    lp.print(out, prefix + "lp.");
}

void LabelPropagationCoarseningContext::print(std::ostream& out, const std::string& prefix) const {
    out << prefix << "num_iterations=" << num_iterations << " "                             //
        << prefix << "max_degree=" << large_degree_threshold << " "                         //
        << prefix << "two_hop_clustering_threshold=" << two_hop_clustering_threshold << " " //
        << prefix << "max_num_neighbors=" << max_num_neighbors << " ";                      //
}

void LabelPropagationRefinementContext::print(std::ostream& out, const std::string& prefix) const {
    out << prefix << "num_iterations=" << num_iterations << " "        //
        << prefix << "max_degree=" << large_degree_threshold << " "    //
        << prefix << "max_num_neighbors=" << max_num_neighbors << " "; //
}

void FMRefinementContext::print(std::ostream& out, const std::string& prefix) const {
    out << prefix << "stopping_rule=" << stopping_rule << " "             //
        << prefix << "num_fruitless_moves=" << num_fruitless_moves << " " //
        << prefix << "alpha=" << alpha << " ";                            //
}

void BalancerRefinementContext::print(std::ostream& out, const std::string& prefix) const {
    out << prefix << "timepoint=" << timepoint << " "  //
        << prefix << "algorithm=" << algorithm << " "; //
}

void RefinementContext::print(std::ostream& out, const std::string& prefix) const {
    out << prefix << "algorithm=" << algorithm << " "; //

    lp.print(out, prefix + "lp.");
    fm.print(out, prefix + "fm.");
    balancer.print(out, prefix + "balancer.");
}

void InitialPartitioningContext::print(std::ostream& out, const std::string& prefix) const {
    coarsening.print(out, prefix + "coarsening.");
    refinement.print(out, prefix + "refinement.");
    out << prefix << "mode=" << mode << " "                                                                 //
        << prefix << "repetition_multiplier=" << repetition_multiplier << " "                               //
        << prefix << "min_num_repetitions=" << min_num_repetitions << " "                                   //
        << prefix << "max_num_repetitions=" << max_num_repetitions << " "                                   //
        << prefix << "num_seed_iterations=" << num_seed_iterations << " "                                   //
        << prefix << "use_adaptive_bipartitioner_selection=" << use_adaptive_bipartitioner_selection << " " //
        << prefix << "multiplier_exponent=" << multiplier_exponent << " ";                                  //
}

void DebugContext::print(std::ostream& out, const std::string& prefix) const {
    out << prefix << "just_sanitize_args=" << just_sanitize_args << " "; //
}

void ParallelContext::print(std::ostream& out, const std::string& prefix) const {
    out << prefix << "use_interleaved_numa_allocation=" << use_interleaved_numa_allocation << " " //
        << prefix << "num_threads=" << num_threads << " ";                                        //
}

void Context::print(std::ostream& out, const std::string& prefix) const {
    out << prefix << "graph_filename=" << graph_filename << " "           //
        << prefix << "seed=" << seed << " "                               //
        << prefix << "save_output_partition=" << save_partition << " "    //
        << prefix << "partition_filename=" << partition_filename << " "   //
        << prefix << "partition_directory=" << partition_directory << " " //
        << prefix << "quiet=" << quiet << " ";                            //

    partition.print(out, prefix + "partition.");
    coarsening.print(out, prefix + "coarsening.");
    initial_partitioning.print(out, prefix + "initial_partitioning.");
    refinement.print(out, prefix + "refinement.");
    debug.print(out, prefix + "debug.");
    parallel.print(out, prefix + "parallel.");
}

void Context::setup(const Graph& graph) {
    partition.setup(graph);
}

Context create_default_context() {
    // clang-format off
  return { // Context
    .graph_filename = "",
    .seed = 0,
    .save_partition = false,
    .partition_directory = "./",
    .partition_filename = "", // generate filename
    .quiet = false,
    .partition = { // Context -> Partition
      .mode = PartitioningMode::DEEP,
      .epsilon = 0.03,
      .k = 2,
      .fast_initial_partitioning = false,
    },
    .coarsening = { // Context -> Coarsening
      .algorithm = ClusteringAlgorithm::LABEL_PROPAGATION,
      .lp = { // Context -> Coarsening -> Label Propagation
        .num_iterations = 5,
        .large_degree_threshold = 1000000,
        .max_num_neighbors = 200000,
        .two_hop_clustering_threshold = 0.5,
      },
      .contraction_limit = 2000,
      .enforce_contraction_limit = false,
      .convergence_threshold = 0.05,
      .cluster_weight_limit = ClusterWeightLimit::EPSILON_BLOCK_WEIGHT,
      .cluster_weight_multiplier = 1.0,
    },
    .initial_partitioning = { // Context -> Initial Partitioning
      .coarsening = { // Context -> Initial Partitioning -> Coarsening
        .algorithm = ClusteringAlgorithm::LABEL_PROPAGATION,
        .lp = { // Context -> Initial Partitioning -> Coarsening -> Label Propagation
          .num_iterations = 1, // no effect
          .large_degree_threshold = 1000000, // no effect
          .max_num_neighbors = 200000, // no effect
          .two_hop_clustering_threshold = 0.5, // no effect
        },
        .contraction_limit = 20,
        .enforce_contraction_limit = false, // no effect
        .convergence_threshold = 0.05,
        .cluster_weight_limit = ClusterWeightLimit::BLOCK_WEIGHT,
        .cluster_weight_multiplier = 1.0 / 12.0,
      },
      .refinement = { // Context -> Initial Partitioning -> Refinement
        .algorithm = RefinementAlgorithm::TWO_WAY_FM,
        .lp = {},
        .fm = { // Context -> Initial Partitioning -> Refinement -> FM
          .stopping_rule = FMStoppingRule::SIMPLE,
          .num_fruitless_moves = 100,
          .alpha = 1.0,
          .num_iterations = 5,
          .improvement_abortion_threshold = 0.0001,
        },
        .balancer = { // Context -> Initial Partitioning -> Refinement -> Balancer
          .algorithm = BalancingAlgorithm::BLOCK_LEVEL_PARALLEL_BALANCER,
          .timepoint = BalancingTimepoint::BEFORE_KWAY_REFINEMENT,
        },
      },
      .mode = InitialPartitioningMode::SYNCHRONOUS_PARALLEL,
      .repetition_multiplier = 1.0,
      .min_num_repetitions = 10,
      .min_num_non_adaptive_repetitions = 5,
      .max_num_repetitions = 50,
      .num_seed_iterations = 1,
      .use_adaptive_bipartitioner_selection = true,
      .multiplier_exponent = 0,
    },
    .refinement = { // Context -> Refinement
      .algorithm = RefinementAlgorithm::LABEL_PROPAGATION,
      .lp = { // Context -> Refinement -> Label Propagation
        .num_iterations = 5,
        .large_degree_threshold = 1000000,
        .max_num_neighbors = std::numeric_limits<NodeID>::max(),
      },
      .fm = {},
      .balancer = { // Context -> Refinement -> Balancer
        .algorithm = BalancingAlgorithm::BLOCK_LEVEL_PARALLEL_BALANCER,
        .timepoint = BalancingTimepoint::BEFORE_KWAY_REFINEMENT,
      },
    },
    .debug = { // Context -> Debug
      .just_sanitize_args = false,
    },
    .parallel = { // Context -> Parallel
      .use_interleaved_numa_allocation = true,
      .num_threads = 1,
    },
  };
    // clang-format on
}

Context create_default_context(const Graph& graph, const BlockID k, const double epsilon) {
    Context context           = create_default_context();
    context.partition.k       = k;
    context.partition.epsilon = epsilon;
    context.setup(graph);
    return context;
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

std::ostream& operator<<(std::ostream& out, const Context& context) {
    context.print(out);
    return out;
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
} // namespace kaminpar::shm
