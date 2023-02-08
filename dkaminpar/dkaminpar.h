/*******************************************************************************
 * @file:   dkaminpar.h
 * @author: Daniel Seemaier
 * @date:   30.01.2023
 * @brief:  Public symbols of the distributed partitioner
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>

#include <mpi.h>
#include <tbb/global_control.h>

#include "kaminpar/context.h"
#include "kaminpar/definitions.h"

namespace kaminpar::mpi {
using PEID = int;
}

// @todo once we build a similar interface for KaMinPar, we can get rid of this forward-declaration
namespace kaminpar::shm {
struct Context;
};

namespace kaminpar::dist {
using GlobalNodeID     = std::uint64_t;
using GlobalNodeWeight = std::int64_t;
using GlobalEdgeID     = std::uint64_t;
using GlobalEdgeWeight = std::int64_t;
using BlockWeight      = std::int64_t;

using mpi::PEID;

using shm::BlockID;
using shm::EdgeID;
using shm::NodeID;

#ifdef KAMINPAR_64BIT_LOCAL_WEIGHTS
using NodeWeight = std::int64_t;
using EdgeWeight = std::int64_t;
#else // KAMINPAR_64BIT_LOCAL_WEIGHTS
using NodeWeight = std::int32_t;
using EdgeWeight = std::int32_t;
#endif

constexpr NodeID           kInvalidNodeID           = std::numeric_limits<NodeID>::max();
constexpr GlobalNodeID     kInvalidGlobalNodeID     = std::numeric_limits<GlobalNodeID>::max();
constexpr NodeWeight       kInvalidNodeWeight       = std::numeric_limits<NodeWeight>::max();
constexpr GlobalNodeWeight kInvalidGlobalNodeWeight = std::numeric_limits<GlobalNodeWeight>::max();
constexpr EdgeID           kInvalidEdgeID           = std::numeric_limits<EdgeID>::max();
constexpr GlobalEdgeID     kInvalidGlobalEdgeID     = std::numeric_limits<GlobalEdgeID>::max();
constexpr EdgeWeight       kInvalidEdgeWeight       = std::numeric_limits<EdgeWeight>::max();
constexpr GlobalEdgeWeight kInvalidGlobalEdgeWeight = std::numeric_limits<GlobalEdgeWeight>::max();
constexpr BlockID          kInvalidBlockID          = std::numeric_limits<BlockID>::max();
constexpr BlockWeight      kInvalidBlockWeight      = std::numeric_limits<BlockWeight>::max();
} // namespace kaminpar::dist

namespace kaminpar::dist {
enum class IODistribution {
    NODE_BALANCED,
    EDGE_BALANCED,
};

enum class IOFormat {
    AUTO,
    TEXT,
    BINARY,
};

enum class OutputLevel : std::uint8_t {
    QUIET,
    PROGRESS,
    APPLICATION,
    EXPERIMENT,
};

enum class PartitioningMode {
    KWAY,
    DEEP,
};

enum class GlobalClusteringAlgorithm {
    NOOP,
    ACTIVE_SET_LP,
    LP,
    LOCKING_LP,
    HEM,
    HEM_LP,
};

enum class LocalClusteringAlgorithm {
    NOOP,
    LP,
};

enum class GlobalContractionAlgorithm {
    NO_MIGRATION,
    MINIMAL_MIGRATION,
    FULL_MIGRATION,
    V2,
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
    COLORED_LP,
    GREEDY_BALANCER,
};

enum class LabelPropagationMoveExecutionStrategy {
    PROBABILISTIC,
    BEST_MOVES,
    LOCAL_MOVES,
};

enum class GraphOrdering {
    NATURAL,
    DEGREE_BUCKETS,
    COLORING,
};

struct ParallelContext {
    std::size_t num_threads = 0;
    std::size_t num_mpis    = 0;
};

struct LabelPropagationCoarseningContext {
    int    num_iterations                       = 0;
    NodeID passive_high_degree_threshold        = 0;
    NodeID active_high_degree_threshold         = 0;
    NodeID max_num_neighbors                    = 0;
    bool   merge_singleton_clusters             = 0;
    double merge_nonadjacent_clusters_threshold = 0;
    int    total_num_chunks                     = 0;
    int    fixed_num_chunks                     = 0;
    int    min_num_chunks                       = 0;
    bool   ignore_ghost_nodes                   = false;
    bool   keep_ghost_clusters                  = false;
    bool   scale_chunks_with_threads            = false;

    bool sync_cluster_weights    = false;
    bool enforce_cluster_weights = false;
    bool cheap_toplevel          = false;

    bool should_merge_nonadjacent_clusters(NodeID old_n, NodeID new_n) const;
    int  compute_num_chunks(const ParallelContext& parallel) const;
};

struct HEMCoarseningContext {
    int    max_num_coloring_chunks            = 0;
    int    fixed_num_coloring_chunks          = 0;
    int    min_num_coloring_chunks            = 0;
    bool   scale_coloring_chunks_with_threads = false;
    double small_color_blacklist              = 0;
    bool   only_blacklist_input_level         = false;
    bool   ignore_weight_limit                = false;

    int compute_num_coloring_chunks(const ParallelContext& parallel) const;
};

struct ColoredLabelPropagationRefinementContext {
    int  num_iterations                  = 0;
    int  num_move_execution_iterations   = 0;
    int  num_probabilistic_move_attempts = 0;
    bool sort_by_rel_gain                = false;

    int    max_num_coloring_chunks            = 0;
    int    fixed_num_coloring_chunks          = 0;
    int    min_num_coloring_chunks            = 0;
    bool   scale_coloring_chunks_with_threads = false;
    double small_color_blacklist              = 0;
    bool   only_blacklist_input_level         = false;

    bool track_local_block_weights = false;
    bool use_active_set            = false;

    LabelPropagationMoveExecutionStrategy move_execution_strategy =
        LabelPropagationMoveExecutionStrategy::PROBABILISTIC;

    int compute_num_coloring_chunks(const ParallelContext& parallel) const;
};

struct LabelPropagationRefinementContext {
    NodeID active_high_degree_threshold = 0;
    int    num_iterations               = 0;

    int total_num_chunks = 0;
    int fixed_num_chunks = 0;
    int min_num_chunks   = 0;

    int  num_move_attempts         = 0;
    bool ignore_probabilities      = false;
    bool scale_chunks_with_threads = false;

    int compute_num_chunks(const ParallelContext& parallel) const;
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
    HEMCoarseningContext              hem;

    std::size_t                       max_local_clustering_levels = 0;
    LocalClusteringAlgorithm          local_clustering_algorithm;
    LabelPropagationCoarseningContext local_lp;

    NodeID                  contraction_limit         = 0;
    shm::ClusterWeightLimit cluster_weight_limit      = shm::ClusterWeightLimit::EPSILON_BLOCK_WEIGHT;
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

struct GreedyBalancerContext {
    NodeID num_nodes_per_block = 0;
};

struct RefinementContext {
    std::vector<KWayRefinementAlgorithm> algorithms;
    bool                                 refine_coarsest_level = false;

    LabelPropagationRefinementContext        lp;
    ColoredLabelPropagationRefinementContext colored_lp;
    FMRefinementContext                      fm;
    GreedyBalancerContext                    greedy_balancer;

    bool includes_algorithm(KWayRefinementAlgorithm algorithm) const;
};

struct PartitionContext {
    PartitionContext(BlockID k, BlockID K, double epsilon);

    PartitionContext(const PartitionContext& other);
    PartitionContext& operator=(const PartitionContext& other);

    ~PartitionContext();

    BlockID k       = kInvalidBlockID;
    BlockID K       = kInvalidBlockID;
    double  epsilon = 0.0;

    std::unique_ptr<struct GraphContext> graph;
};

struct DebugContext {
    std::string graph_filename                  = "";
    bool        save_finest_graph               = false;
    bool        save_coarsest_graph             = false;
    bool        save_graph_hierarchy            = false;
    bool        save_clustering_hierarchy       = false;
    bool        save_partition_hierarchy        = false;
    bool        save_unrefined_finest_partition = false;
};

struct Context {
    GraphOrdering rearrange_by = GraphOrdering::NATURAL;

    PartitioningMode mode = PartitioningMode::DEEP;

    bool enable_pe_splitting   = false;
    bool simulate_singlethread = false;

    PartitionContext           partition;
    ParallelContext            parallel;
    CoarseningContext          coarsening;
    InitialPartitioningContext initial_partitioning;
    RefinementContext          refinement;
    DebugContext               debug;
};
} // namespace kaminpar::dist

namespace kaminpar::dist {
Context create_context_by_preset_name(const std::string& name);

Context create_default_context();
Context create_strong_context();
Context create_default_social_context();

std::unordered_set<std::string> get_preset_names();

struct GraphPtr {
    GraphPtr();
    GraphPtr(std::unique_ptr<class DistributedGraph> graph);

    GraphPtr(const GraphPtr&)            = delete;
    GraphPtr& operator=(const GraphPtr&) = delete;

    GraphPtr(GraphPtr&&) noexcept;
    GraphPtr& operator=(GraphPtr&&) noexcept;

    ~GraphPtr();

    std::unique_ptr<class DistributedGraph> ptr;
};

class DistributedGraphPartitioner {
public:
    DistributedGraphPartitioner(MPI_Comm comm, int num_threads, Context ctx);

    void set_output_level(OutputLevel output_level);

    void set_max_timer_depth(int max_timer_depth);

    Context& context();

    void import_graph(
        GlobalNodeID* node_distribution, GlobalEdgeID* nodes, GlobalNodeID* edges, GlobalNodeWeight* node_weights,
        GlobalEdgeWeight* edge_weights
    );

    NodeID load_graph(const std::string& filename, IOFormat format, IODistribution distribution);

    GlobalEdgeWeight compute_partition(int seed, BlockID k, BlockID *partition);

private:
    MPI_Comm _comm;
    int      _num_threads;

    int         _max_timer_depth = std::numeric_limits<int>::max();
    OutputLevel _output_level    = OutputLevel::APPLICATION;
    Context     _ctx;

    GraphPtr            _graph_ptr;
    tbb::global_control _gc;

    bool _was_rearranged = false;
};
} // namespace kaminpar::dist
