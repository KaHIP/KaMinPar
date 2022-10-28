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

struct ParallelContext {
    std::size_t num_threads                     = 0;
    std::size_t num_mpis                        = 0;
    bool        use_interleaved_numa_allocation = false;
    int         mpi_thread_support              = false;
    bool        simulate_singlethread           = false;
};

struct LabelPropagationCoarseningContext {
    std::size_t num_iterations                       = 0;
    NodeID      passive_high_degree_threshold        = 0;
    NodeID      active_high_degree_threshold         = 0;
    NodeID      max_num_neighbors                    = 0;
    bool        merge_singleton_clusters             = 0;
    double      merge_nonadjacent_clusters_threshold = 0;
    std::size_t total_num_chunks                     = 0;
    std::size_t num_chunks                           = 0;
    std::size_t min_num_chunks                       = 0;
    bool        ignore_ghost_nodes                   = false;
    bool        keep_ghost_clusters                  = false;
    bool        scale_chunks_with_threads            = false;

    [[nodiscard]] bool should_merge_nonadjacent_clusters(NodeID old_n, NodeID new_n) const;
    void               setup(const ParallelContext& parallel);
};

struct LabelPropagationRefinementContext {
    NodeID      active_high_degree_threshold = 0;
    std::size_t num_iterations               = 0;
    std::size_t total_num_chunks             = 0;
    std::size_t num_chunks                   = 0;
    std::size_t min_num_chunks               = 0;
    std::size_t num_move_attempts            = 0;
    bool        ignore_probabilities         = false;
    bool        scale_chunks_with_threads    = false;

    void setup(const ParallelContext& parallel);
};

struct FMRefinementContext {
    double      alpha           = 0.0;
    NodeID      radius          = 0;
    PEID        pe_radius       = 0;
    bool        overlap_regions = false;
    std::size_t num_iterations  = 0;
    bool        sequential      = false;
    bool        premove_locally = false;
    NodeID      bound_degree    = 0;
    bool        contract_border = false;
};

struct CoarseningContext {
    std::size_t                       max_global_clustering_levels = 0;
    GlobalClusteringAlgorithm         global_clustering_algorithm;
    GlobalContractionAlgorithm        global_contraction_algorithm;
    LabelPropagationCoarseningContext global_lp;

    std::size_t                       max_local_clustering_levels = 0;
    LocalClusteringAlgorithm          local_clustering_algorithm;
    LabelPropagationCoarseningContext local_lp;

    NodeID                  contraction_limit = 0;
    shm::ClusterWeightLimit cluster_weight_limit;
    double                  cluster_weight_multiplier = 0.0;

    void setup(const ParallelContext& parallel);
};

struct MtKaHyParContext {
    std::string preset_filename = "";
};

struct InitialPartitioningContext {
    InitialPartitioningAlgorithm algorithm;
    MtKaHyParContext             mtkahypar;
    shm::Context                 kaminpar;
};

struct BalancingContext {
    BalancingAlgorithm algorithm;
    NodeID             num_nodes_per_block = 0;
};

struct RefinementContext {
    KWayRefinementAlgorithm           algorithm;
    LabelPropagationRefinementContext lp;
    FMRefinementContext               fm;
    BalancingContext                  balancing;
    bool                              refine_coarsest_level = false;

    void setup(const ParallelContext& parallel);
};

struct PartitionContext;

struct GraphContext {
public:
    GraphContext() = default;
    GraphContext(const DistributedGraph& graph, const PartitionContext& p_ctx);
    GraphContext(const shm::Graph& graph, const PartitionContext& p_ctx);

    [[nodiscard]] bool initialized() const {
        return _global_n != kInvalidGlobalNodeID;
    }

    [[nodiscard]] GlobalNodeID global_n() const {
        KASSERT(_global_n != kInvalidGlobalNodeID);
        return _global_n;
    }

    [[nodiscard]] NodeID n() const {
        KASSERT(_n != kInvalidNodeID);
        return _n;
    }

    [[nodiscard]] NodeID total_n() const {
        KASSERT(_total_n != kInvalidNodeID);
        return _total_n;
    }

    [[nodiscard]] GlobalEdgeID global_m() const {
        KASSERT(_global_m != kInvalidGlobalEdgeID);
        return _global_m;
    }

    [[nodiscard]] EdgeID m() const {
        KASSERT(_m != kInvalidEdgeID);
        return _m;
    }

    [[nodiscard]] NodeWeight total_node_weight() const {
        KASSERT(_total_node_weight != kInvalidNodeWeight);
        return _total_node_weight;
    }

    [[nodiscard]] GlobalNodeWeight global_total_node_weight() const {
        KASSERT(_global_total_node_weight != kInvalidGlobalNodeWeight);
        return _global_total_node_weight;
    }

    [[nodiscard]] EdgeWeight total_edge_weight() const {
        KASSERT(_total_edge_weight != kInvalidEdgeWeight);
        return _total_edge_weight;
    }

    [[nodiscard]] GlobalEdgeWeight global_total_edge_weight() const {
        KASSERT(_global_total_edge_weight != kInvalidEdgeWeight);
        return _global_total_edge_weight;
    }

    [[nodiscard]] const auto& perfectly_balanced_block_weights() const {
        KASSERT(!_perfectly_balanced_block_weights.empty());
        return _perfectly_balanced_block_weights;
    }

    [[nodiscard]] BlockWeight perfectly_balanced_block_weight(const BlockID b) const {
        KASSERT(b < _perfectly_balanced_block_weights.size());
        return _perfectly_balanced_block_weights[b];
    }

    [[nodiscard]] const auto& max_block_weights() const {
        KASSERT(!_max_block_weights.empty());
        return _max_block_weights;
    }

    [[nodiscard]] BlockWeight max_block_weight(const BlockID b) const {
        KASSERT(b < _max_block_weights.size());
        return _max_block_weights[b];
    }

private:
    void setup_perfectly_balanced_block_weights(const BlockID k);
    void setup_max_block_weights(const BlockID k, const double epsilon);

    GlobalNodeID     _global_n                 = kInvalidGlobalNodeID;
    NodeID           _n                        = kInvalidNodeID;
    NodeID           _total_n                  = kInvalidNodeID;
    GlobalEdgeID     _global_m                 = kInvalidGlobalEdgeID;
    EdgeID           _m                        = kInvalidEdgeID;
    GlobalNodeWeight _global_total_node_weight = kInvalidGlobalNodeID;
    NodeWeight       _total_node_weight        = kInvalidNodeWeight;
    GlobalNodeWeight _global_max_node_weight   = kInvalidNodeWeight;
    GlobalEdgeWeight _global_total_edge_weight = kInvalidEdgeWeight;
    EdgeWeight       _total_edge_weight        = kInvalidEdgeWeight;

    NoinitVector<BlockWeight> _perfectly_balanced_block_weights{};
    NoinitVector<BlockWeight> _max_block_weights{};
};

struct PartitionContext {
    BlockID          k                   = kInvalidBlockID;
    BlockID          K                   = kInvalidBlockID;
    double           epsilon             = 0.0;
    PartitioningMode mode                = PartitioningMode::DEEP;
    bool             enable_pe_splitting = false;

    GraphContext graph;

    void setup(const DistributedGraph& graph);
    void setup(const shm::Graph& graph);
};

struct DebugContext {
    bool save_imbalanced_partitions = false;
    bool save_graph_hierarchy       = false;
    bool save_coarsest_graph        = false;
    bool save_clustering_hierarchy  = false;
};

struct Context {
    std::string graph_filename     = "";
    bool        load_edge_balanced = false;
    int         seed               = 0;
    bool        quiet              = false;
    std::size_t num_repetitions    = 1;
    std::size_t time_limit         = 0;
    bool        sort_graph         = false;
    bool        parsable_output    = false;
    int         timer_depth        = 3;

    PartitionContext           partition;
    ParallelContext            parallel;
    CoarseningContext          coarsening;
    InitialPartitioningContext initial_partitioning;
    RefinementContext          refinement;
    DebugContext               debug;

    void setup(const DistributedGraph& graph);
};
} // namespace kaminpar::dist
