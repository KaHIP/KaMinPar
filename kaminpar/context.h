/*******************************************************************************
 * @file:   context.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:  Configuration struct for KaMinPar.
 ******************************************************************************/
#pragma once

#include <cmath>
#include <map>
#include <string_view>

#include "kaminpar/datastructure/graph.h"
#include "kaminpar/definitions.h"

#include "common/utils/enum_string_conversion.h"

namespace kaminpar::shm {
enum class ClusteringAlgorithm {
    NOOP,
    LABEL_PROPAGATION,
};

enum class ClusterWeightLimit {
    EPSILON_BLOCK_WEIGHT,
    BLOCK_WEIGHT,
    ONE,
    ZERO,
};

enum class RefinementAlgorithm {
    LABEL_PROPAGATION,
    TWO_WAY_FM,
    NOOP,
};

enum class FMStoppingRule {
    SIMPLE,
    ADAPTIVE,
};

enum class BalancingTimepoint {
    BEFORE_KWAY_REFINEMENT,
    AFTER_KWAY_REFINEMENT,
    ALWAYS,
    NEVER,
};

enum class BalancingAlgorithm {
    NOOP,
    BLOCK_LEVEL_PARALLEL_BALANCER,
};

enum class PartitioningMode {
    DEEP,
    RB,
};

enum class InitialPartitioningMode {
    SEQUENTIAL,
    ASYNCHRONOUS_PARALLEL,
    SYNCHRONOUS_PARALLEL,
};

std::ostream& operator<<(std::ostream& out, ClusteringAlgorithm algorithm);
std::ostream& operator<<(std::ostream& out, ClusterWeightLimit limit);
std::ostream& operator<<(std::ostream& out, RefinementAlgorithm algorithm);
std::ostream& operator<<(std::ostream& out, FMStoppingRule rule);
std::ostream& operator<<(std::ostream& out, BalancingTimepoint timepoint);
std::ostream& operator<<(std::ostream& out, BalancingAlgorithm algorithm);
std::ostream& operator<<(std::ostream& out, PartitioningMode mode);
std::ostream& operator<<(std::ostream& out, InitialPartitioningMode mode);

std::unordered_map<std::string, ClusteringAlgorithm>     get_clustering_algorithms();
std::unordered_map<std::string, ClusterWeightLimit>      get_cluster_weight_limits();
std::unordered_map<std::string, RefinementAlgorithm>     get_2way_refinement_algorithms();
std::unordered_map<std::string, RefinementAlgorithm>     get_kway_refinement_algorithms();
std::unordered_map<std::string, FMStoppingRule>          get_fm_stopping_rules();
std::unordered_map<std::string, BalancingTimepoint>      get_balancing_timepoints();
std::unordered_map<std::string, BalancingAlgorithm>      get_balancing_algorithms();
std::unordered_map<std::string, PartitioningMode>        get_partitioning_modes();
std::unordered_map<std::string, InitialPartitioningMode> get_initial_partitioning_modes();

struct PartitionContext;

struct BlockWeightsContext {
    void setup(const PartitionContext& ctx);
    void setup(const PartitionContext& ctx, const scalable_vector<BlockID>& final_ks);

    [[nodiscard]] BlockWeight                         max(BlockID b) const;
    [[nodiscard]] const scalable_vector<BlockWeight>& all_max() const;
    [[nodiscard]] BlockWeight                         perfectly_balanced(BlockID b) const;
    [[nodiscard]] const scalable_vector<BlockWeight>& all_perfectly_balanced() const;

private:
    scalable_vector<BlockWeight> _perfectly_balanced_block_weights;
    scalable_vector<BlockWeight> _max_block_weights;
};

struct PartitionContext {
    PartitioningMode mode;                      //! Partitioning mode, e.g., k-way, rb, deep MGP
    double           epsilon;                   //! Imbalance factor.
    BlockID          k;                         //! Number of blocks.

    //! Balance constraint: precomputed maximum block weights.
    BlockWeightsContext block_weights{};

    void setup(const Graph& graph);
    void setup_block_weights();

    NodeID     n                 = kInvalidNodeID;     //! Number of nodes in the input graph.
    EdgeID     m                 = kInvalidEdgeID;     //! Number of edges in the input graph.
    NodeWeight total_node_weight = kInvalidNodeWeight; //! Total node weight in the input graph.
    EdgeWeight total_edge_weight = kInvalidEdgeWeight; //! Total edge weight in the input graph.
    NodeWeight max_node_weight   = kInvalidNodeWeight; //! Weight of heaviest node in the input graph.

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct LabelPropagationCoarseningContext {
    std::size_t num_iterations;               //! Maximum number of LP iterations.
    Degree      large_degree_threshold;       //! Ignore nodes with degree larger than this.
    NodeID      max_num_neighbors;            //! Only consider this many neighbors of each node.
    double      two_hop_clustering_threshold; //! Perform 2-hop clustering if graph shrunk by less than this factor.

    void print(std::ostream& out, const std::string& prefix = "") const;

    [[nodiscard]] bool use_two_hop_clustering(const NodeID old_n, const NodeID new_n) const {
        return (1.0 - 1.0 * new_n / old_n) <= two_hop_clustering_threshold;
    }
};

struct CoarseningContext {
    ClusteringAlgorithm               algorithm; //! Clustering algorithm.
    LabelPropagationCoarseningContext lp;        //! Configuration for LP as clustering algorithm.
    NodeID contraction_limit;         //! Abort coarsening if the coarsest graph has less than twice this many nodes.
    bool   enforce_contraction_limit; //! Force the clustering algorithm to converge on the contraction limit.
    double convergence_threshold;     //! Coarsening converges if the graph shrunk by less than this factor.
    ClusterWeightLimit cluster_weight_limit;      //! Rule to compute the maximum cluster weight.
    double             cluster_weight_multiplier; //! Multiplicative factor to the maximum cluster weight.

    void print(std::ostream& out, const std::string& prefix = "") const;

    [[nodiscard]] inline bool coarsening_should_converge(const NodeID old_n, const NodeID new_n) const {
        return (1.0 - 1.0 * new_n / old_n) <= convergence_threshold;
    }
};

struct LabelPropagationRefinementContext {
    std::size_t num_iterations;         //! Maximum number of LP iterations.
    Degree      large_degree_threshold; //! Ignore nodes with degree larger than this.
    NodeID      max_num_neighbors;      //! Only consider this many neighbors of each node.

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct FMRefinementContext {
    FMStoppingRule stopping_rule;          //! Rule to determine when to stop FM.
    NodeID         num_fruitless_moves;    //! Stop after more than this many moves without cut improvement.
    double         alpha;                  //! Config parameter of the adaptive stopping criteria.
    std::size_t    num_iterations;         //! Maximum number of FM iterations.
    double improvement_abortion_threshold; //! Stop FM if last iteration improved the cut by less than this factor.

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct BalancerRefinementContext {
    BalancingAlgorithm algorithm; //! Balancing algorithm.
    BalancingTimepoint timepoint; //! Rule to determine when to run the balancing algorithm.

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct RefinementContext {
    RefinementAlgorithm               algorithm; //! Refinement algorithm.
    LabelPropagationRefinementContext lp;        //! If LP is used: configuration for LP.
    FMRefinementContext               fm;        //! If FM is used: configuration for FM.
    BalancerRefinementContext         balancer;  //! If a balancer is used: configuration for balancer.

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct InitialPartitioningContext {
    CoarseningContext       coarsening; //! Configuration for IP coarsening.
    RefinementContext       refinement; //! Configuration for IP refinement.
    InitialPartitioningMode mode;       //! Initial partitioning mode.
    double                  repetition_multiplier;
    std::size_t             min_num_repetitions;
    std::size_t             min_num_non_adaptive_repetitions;
    std::size_t             max_num_repetitions;
    std::size_t             num_seed_iterations;
    bool                    use_adaptive_bipartitioner_selection;
    std::size_t             multiplier_exponent;

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct DebugContext {
    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct ParallelContext {
    bool        use_interleaved_numa_allocation;
    std::size_t num_threads;

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct Context {
    std::string graph_filename;
    int         seed;
    bool        save_partition;
    std::string partition_directory;
    std::string partition_filename;
    bool        quiet;

    PartitionContext           partition;
    CoarseningContext          coarsening;
    InitialPartitioningContext initial_partitioning;
    RefinementContext          refinement;
    DebugContext               debug;
    ParallelContext            parallel;

    void print(std::ostream& out, const std::string& prefix = "") const;

    void setup(const Graph& graph);

    [[nodiscard]] std::string partition_file() const {
        return partition_directory + "/" + partition_filename;
    }
};

std::ostream& operator<<(std::ostream& out, const Context& context);

PartitionContext
create_bipartition_context(const PartitionContext& k_p_ctx, const Graph& subgraph, BlockID final_k1, BlockID final_k2);

double compute_2way_adaptive_epsilon(
    const PartitionContext& p_ctx, NodeWeight subgraph_total_node_weight, BlockID subgraph_final_k
);

template <typename NodeID_ = NodeID, typename NodeWeight_ = NodeWeight>
NodeWeight_ compute_max_cluster_weight(
    const NodeID_ n, const NodeWeight_ total_node_weight, const PartitionContext& input_p_ctx,
    const CoarseningContext& c_ctx
) {
    double max_cluster_weight = 0.0;

    switch (c_ctx.cluster_weight_limit) {
        case ClusterWeightLimit::EPSILON_BLOCK_WEIGHT:
            max_cluster_weight = (input_p_ctx.epsilon * total_node_weight)
                                 / std::clamp<BlockID>(n / c_ctx.contraction_limit, 2, input_p_ctx.k);
            break;

        case ClusterWeightLimit::BLOCK_WEIGHT:
            max_cluster_weight = (1.0 + input_p_ctx.epsilon) * total_node_weight / input_p_ctx.k;
            break;

        case ClusterWeightLimit::ONE:
            max_cluster_weight = 1.0;
            break;

        case ClusterWeightLimit::ZERO:
            max_cluster_weight = 0.0;
            break;
    }

    return static_cast<NodeWeight_>(max_cluster_weight * c_ctx.cluster_weight_multiplier);
}

template <typename NodeWeight_ = NodeWeight, typename Graph_ = Graph>
NodeWeight_
compute_max_cluster_weight(const Graph_& c_graph, const PartitionContext& input_p_ctx, const CoarseningContext& c_ctx) {
    return compute_max_cluster_weight(c_graph.n(), c_graph.total_node_weight(), input_p_ctx, c_ctx);
}
} // namespace kaminpar::shm
