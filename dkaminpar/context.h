/*******************************************************************************
 * @file:   context.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Context struct for the distributed graph partitioner.
 ******************************************************************************/
#pragma once

#include "dkaminpar/datastructures/distributed_graph.h"
#include "dkaminpar/definitions.h"

#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"

namespace kaminpar::dist {
enum class PartitioningMode {
    KWAY,
    DEEP,
    DEEPER,
};

enum class GlobalClusteringAlgorithm {
    NOOP,
    ACTIVE_SET_LP,
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
    MTKAHYPAR,
    RANDOM,
};

enum class KWayRefinementAlgorithm {
    NOOP,
    LP,
    LOCAL_FM,
    FM,
    LP_THEN_LOCAL_FM,
    LP_THEN_FM,
};

enum class BalancingAlgorithm {
    DISTRIBUTED,
};

std::ostream& operator<<(std::ostream& out, PartitioningMode mode);
std::ostream& operator<<(std::ostream& out, GlobalClusteringAlgorithm algorithm);
std::ostream& operator<<(std::ostream& out, LocalClusteringAlgorithm algorithm);
std::ostream& operator<<(std::ostream& out, GlobalContractionAlgorithm algorithm);
std::ostream& operator<<(std::ostream& out, InitialPartitioningAlgorithm algorithm);
std::ostream& operator<<(std::ostream& out, KWayRefinementAlgorithm algorithm);
std::ostream& operator<<(std::ostream& out, BalancingAlgorithm algorithm);

std::unordered_map<std::string, PartitioningMode>             get_partitioning_modes();
std::unordered_map<std::string, GlobalClusteringAlgorithm>    get_global_clustering_algorithms();
std::unordered_map<std::string, LocalClusteringAlgorithm>     get_local_clustering_algorithms();
std::unordered_map<std::string, GlobalContractionAlgorithm>   get_global_contraction_algorithms();
std::unordered_map<std::string, InitialPartitioningAlgorithm> get_initial_partitioning_algorithms();
std::unordered_map<std::string, KWayRefinementAlgorithm>      get_kway_refinement_algorithms();
std::unordered_map<std::string, BalancingAlgorithm>           get_balancing_algorithms();

struct ParallelContext {
    std::size_t num_threads;
    std::size_t num_mpis;
    bool        use_interleaved_numa_allocation;
    int         mpi_thread_support;
    bool        simulate_singlethread;

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct LabelPropagationCoarseningContext {
    std::size_t num_iterations;
    NodeID      passive_high_degree_threshold;
    NodeID      active_high_degree_threshold;
    NodeID      max_num_neighbors;
    bool        merge_singleton_clusters;
    double      merge_nonadjacent_clusters_threshold;
    std::size_t total_num_chunks;
    std::size_t num_chunks;
    std::size_t min_num_chunks;
    bool        ignore_ghost_nodes;
    bool        keep_ghost_clusters;
    bool        scale_chunks_with_threads;

    [[nodiscard]] bool should_merge_nonadjacent_clusters(const NodeID old_n, const NodeID new_n) const {
        return (1.0 - 1.0 * static_cast<double>(new_n) / static_cast<double>(old_n))
               <= merge_nonadjacent_clusters_threshold;
    }

    void setup(const ParallelContext& parallel) {
        if (num_chunks == 0) {
            const std::size_t chunks =
                scale_chunks_with_threads ? total_num_chunks / parallel.num_threads : total_num_chunks;
            num_chunks = std::max<std::size_t>(8, chunks / parallel.num_mpis);
        }
    }

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct LabelPropagationRefinementContext {
    NodeID      active_high_degree_threshold;
    std::size_t num_iterations;
    std::size_t total_num_chunks;
    std::size_t num_chunks;
    std::size_t min_num_chunks;
    std::size_t num_move_attempts;
    bool        ignore_probabilities;
    bool        scale_chunks_with_threads;

    void setup(const ParallelContext& parallel) {
        if (num_chunks == 0) {
            const std::size_t chunks =
                scale_chunks_with_threads ? total_num_chunks / parallel.num_threads : total_num_chunks;
            num_chunks = std::max<std::size_t>(8, chunks / parallel.num_mpis);
        }
    }

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct FMRefinementContext {
    double      alpha;
    NodeID      radius;
    PEID        pe_radius;
    bool        overlap_regions;
    std::size_t num_iterations;
    bool        sequential;
    bool        premove_locally;
    NodeID      bound_degree;
    bool        contract_border;

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

    void setup(const ParallelContext& parallel) {
        local_lp.setup(parallel);
        global_lp.setup(parallel);
    }

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct MtKaHyParContext {
    std::string preset_filename;

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct InitialPartitioningContext {
    InitialPartitioningAlgorithm algorithm;
    MtKaHyParContext             mtkahypar;
    shm::Context                 kaminpar;

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
    FMRefinementContext               fm;
    BalancingContext                  balancing;
    bool                              refine_coarsest_level;

    void setup(const ParallelContext& parallel) {
        lp.setup(parallel);
    }

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct PartitionContext {
    PartitionContext() = default;

    // required for braces-initializer with private members
    PartitionContext(const BlockID k, const BlockID k_prime, const double epsilon, const PartitioningMode mode)
        : k{k},
          k_prime{k_prime},
          epsilon{epsilon},
          mode{mode} {}

    BlockID          k{};
    BlockID          k_prime{};
    double           epsilon{};
    PartitioningMode mode{};

    void setup(const DistributedGraph& graph);
    void setup(const shm::Graph& graph);

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

struct DebugContext {
    bool save_imbalanced_partitions;
    bool save_graph_hierarchy;
    bool save_coarsest_graph;
    bool save_clustering_hierarchy;

    void print(std::ostream& out, const std::string& prefix = "") const;
};

struct Context {
    std::string graph_filename{};
    bool        load_edge_balanced{};
    int         seed{0};
    bool        quiet{};
    std::size_t num_repetitions;
    std::size_t time_limit;
    bool        sort_graph;
    bool        parsable_output;

    PartitionContext           partition;
    ParallelContext            parallel;
    CoarseningContext          coarsening;
    InitialPartitioningContext initial_partitioning;
    RefinementContext          refinement;
    DebugContext               debug;

    void setup(const DistributedGraph& graph) {
        coarsening.setup(parallel);
        refinement.setup(parallel);
        partition.setup(graph);
    }

    void print(std::ostream& out, const std::string& prefix = "") const;
};

std::ostream& operator<<(std::ostream& out, const Context& context);
} // namespace kaminpar::dist
