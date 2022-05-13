/*******************************************************************************
 * @file:   distributed_context.h
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/definitions.h"
#include "kaminpar/context.h"

namespace dkaminpar {
enum class PartitioningMode {
    KWAY,
    RB,
    DEEP,
};

enum class GlobalClusteringAlgorithm {
    NOOP,
    LP,
    LOCKING_LP,
};

enum class LocalClusteringAlgorithm {
    NOOP,
    LP,
};

enum class GlobalContractionAlgorithm {
    NO_MIGRATION,
    MINIMAL_MIGRATION,
    FULL_MIGRATION,
};

enum class InitialPartitioningAlgorithm {
    KAMINPAR,
    RANDOM,
};

enum class KWayRefinementAlgorithm {
    NOOP,
    PROB_LP,
};

enum class BalancingAlgorithm {
    DISTRIBUTED,
};

DECLARE_ENUM_STRING_CONVERSION(PartitioningMode, partitioning_mode);
DECLARE_ENUM_STRING_CONVERSION(GlobalContractionAlgorithm, global_contraction_algorithm);
DECLARE_ENUM_STRING_CONVERSION(GlobalClusteringAlgorithm, global_clustering_algorithm);
DECLARE_ENUM_STRING_CONVERSION(LocalClusteringAlgorithm, local_clustering_algorithm);
DECLARE_ENUM_STRING_CONVERSION(InitialPartitioningAlgorithm, initial_partitioning_algorithm);
DECLARE_ENUM_STRING_CONVERSION(KWayRefinementAlgorithm, kway_refinement_algorithm);
DECLARE_ENUM_STRING_CONVERSION(BalancingAlgorithm, balancing_algorithm);

struct LabelPropagationCoarseningContext {
    std::size_t num_iterations;
    NodeID      large_degree_threshold;
    NodeID      max_num_neighbors;
    bool        merge_singleton_clusters;
    double      merge_nonadjacent_clusters_threshold;
    std::size_t total_num_chunks;
    std::size_t num_chunks;
    std::size_t min_num_chunks;

    [[nodiscard]] bool should_merge_nonadjacent_clusters(const NodeID old_n, const NodeID new_n) const {
        return (1.0 - 1.0 * static_cast<double>(new_n) / static_cast<double>(old_n))
               <= merge_nonadjacent_clusters_threshold;
    }

    void setup(const DistributedGraph& graph) {
        if (num_chunks == 0) {
            num_chunks = std::max<std::size_t>(8, total_num_chunks / mpi::get_comm_size(graph.communicator()));
        }
    }

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct LabelPropagationRefinementContext {
    std::size_t num_iterations;
    std::size_t total_num_chunks;
    std::size_t num_chunks;
    std::size_t min_num_chunks;
    std::size_t num_move_attempts;

    void setup(const DistributedGraph& graph) {
        if (num_chunks == 0) {
            num_chunks = std::max<std::size_t>(8, total_num_chunks / mpi::get_comm_size(graph.communicator()));
        }
    }

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct CoarseningContext {
    std::size_t                       max_global_clustering_levels;
    GlobalClusteringAlgorithm         global_clustering_algorithm;
    GlobalContractionAlgorithm        global_contraction_algorithm;
    LabelPropagationCoarseningContext global_lp;

    std::size_t                       max_local_clustering_levels;
    LocalClusteringAlgorithm          local_clustering_algorithm;
    LabelPropagationCoarseningContext local_lp;

    NodeID                  contraction_limit;
    shm::ClusterWeightLimit cluster_weight_limit;
    double                  cluster_weight_multiplier;

    void setup(const DistributedGraph& graph) {
        local_lp.setup(graph);
        global_lp.setup(graph);
    }

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct InitialPartitioningContext {
    InitialPartitioningAlgorithm algorithm;
    shm::Context                 sequential;

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct BalancingContext {
    BalancingAlgorithm algorithm;
    NodeID             num_nodes_per_block;

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct RefinementContext {
    KWayRefinementAlgorithm           algorithm;
    LabelPropagationRefinementContext lp;
    BalancingContext                  balancing;

    void setup(const DistributedGraph& graph) {
        lp.setup(graph);
    }

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct ParallelContext {
    std::size_t num_threads;
    bool        use_interleaved_numa_allocation;
    int         mpi_thread_support;

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct PartitionContext {
    // required for braces-initializer with private members
    PartitionContext(const BlockID k, const double epsilon, const PartitioningMode mode)
        : k{k},
          epsilon{epsilon},
          mode{mode} {}

    BlockID          k{};
    double           epsilon{};
    PartitioningMode mode{};

    void setup(const DistributedGraph& graph);

    [[nodiscard]] GlobalNodeID global_n() const {
        KASSERT(_global_n != kInvalidGlobalNodeID);
        return _global_n;
    }

    [[nodiscard]] GlobalEdgeID global_m() const {
        KASSERT(_global_m != kInvalidGlobalEdgeID);
        return _global_m;
    }

    [[nodiscard]] GlobalNodeWeight global_total_node_weight() const {
        KASSERT(_global_total_node_weight != kInvalidGlobalNodeWeight);
        return _global_total_node_weight;
    }

    [[nodiscard]] NodeID local_n() const {
        KASSERT(_local_n != kInvalidNodeID);
        return _local_n;
    }

    [[nodiscard]] NodeID total_n() const {
        KASSERT(_total_n != kInvalidNodeID);
        return _total_n;
    }

    [[nodiscard]] EdgeID local_m() const {
        KASSERT(_local_m != kInvalidEdgeID);
        return _local_m;
    }

    [[nodiscard]] NodeWeight total_node_weight() const {
        KASSERT(_total_node_weight != kInvalidNodeWeight);
        return _total_node_weight;
    }

    [[nodiscard]] inline BlockWeight perfectly_balanced_block_weight(const BlockID b) const {
        KASSERT(b < _perfectly_balanced_block_weights.size());
        return _perfectly_balanced_block_weights[b];
    }

    [[nodiscard]] inline BlockWeight max_block_weight(const BlockID b) const {
        KASSERT(b < _max_block_weights.size());
        return _max_block_weights[b];
    }

    [[nodiscard]] inline const auto& max_block_weights() const {
        return _max_block_weights;
    }

    void print(std::ostream& out, const std::string& prefix = "") const;

private:
    void setup_perfectly_balanced_block_weights();
    void setup_max_block_weights();

    GlobalNodeID     _global_n{kInvalidGlobalNodeID};
    GlobalEdgeID     _global_m{kInvalidGlobalEdgeID};
    GlobalNodeWeight _global_total_node_weight{kInvalidGlobalNodeWeight};
    NodeID           _local_n{kInvalidNodeID};
    EdgeID           _local_m{kInvalidEdgeID};
    NodeID           _total_n{kInvalidNodeID};
    NodeWeight       _total_node_weight{kInvalidNodeWeight};
    NodeWeight       _global_max_node_weight{kInvalidNodeWeight};

    scalable_vector<BlockWeight> _perfectly_balanced_block_weights{};
    scalable_vector<BlockWeight> _max_block_weights{};
};

struct Context {
    std::string graph_filename{};
    bool        load_edge_balanced{};
    int         seed{0};
    bool        quiet{};

    bool save_imbalanced_partitions;
    bool save_coarsest_graph;

    PartitionContext           partition;
    ParallelContext            parallel;
    CoarseningContext          coarsening;
    InitialPartitioningContext initial_partitioning;
    RefinementContext          refinement;

    void setup(const DistributedGraph& graph) {
        coarsening.setup(graph);
        refinement.setup(graph);
        partition.setup(graph);
    }

    void print(std::ostream& out, const std::string& prefix = "") const;
};

std::ostream& operator<<(std::ostream& out, const Context& context);

Context create_default_context();
} // namespace dkaminpar
