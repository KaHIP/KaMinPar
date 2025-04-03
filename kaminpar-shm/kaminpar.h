/*******************************************************************************
 * Public library interface of KaMinPar.
 *
 * @file:   kaminpar.h
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 ******************************************************************************/
#pragma once

#ifdef __cplusplus
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include <tbb/global_control.h>
#endif // __cplusplus

#include <stdbool.h>
#include <stdint.h>

#define KAMINPAR_VERSION_MAJOR 3
#define KAMINPAR_VERSION_MINOR 2
#define KAMINPAR_VERSION_PATCH 0

#ifdef __cplusplus
namespace kaminpar {

enum class OutputLevel : std::uint8_t {
  QUIET,       //! Disable all output to stdout.
  PROGRESS,    //! Continuously output progress information while partitioning.
  APPLICATION, //! Also output the application banner and context summary.
  EXPERIMENT,  //! Also output information only relevant for benchmarking.
  DEBUG,       //! Also output (a sane amount) of debug information.
};

} // namespace kaminpar
#endif // __cplusplus

// C interface:
typedef enum {
  KAMINPAR_OUTPUT_LEVEL_QUIET = 0,
  KAMINPAR_OUTPUT_LEVEL_PROGRESS = 1,
  KAMINPAR_OUTPUT_LEVEL_APPLICATION = 2,
  KAMINPAR_OUTPUT_LEVEL_EXPERIMENT = 3,
  KAMINPAR_OUTPUT_LEVEL_DEBUG = 4,
} kaminpar_output_level_t;

#ifdef __cplusplus
namespace kaminpar::shm {

#ifdef KAMINPAR_64BIT_NODE_IDS
using NodeID = std::uint64_t;
#else  // KAMINPAR_64BIT_NODE_IDS
using NodeID = std::uint32_t;
#endif // KAMINPAR_64BIT_NODE_IDS

#ifdef KAMINPAR_64BIT_EDGE_IDS
using EdgeID = std::uint64_t;
#else  // KAMINPAR_64BIT_EDGE_IDS
using EdgeID = std::uint32_t;
#endif // KAMINPAR_64BIT_EDGE_IDS

#ifdef KAMINPAR_64BIT_WEIGHTS
using NodeWeight = std::int64_t;
using EdgeWeight = std::int64_t;
using UnsignedEdgeWeight = std::uint64_t;
using UnsignedNodeWeight = std::uint64_t;
#else  // KAMINPAR_64BIT_WEIGHTS
using NodeWeight = std::int32_t;
using UnsignedNodeWeight = std::uint32_t;
using EdgeWeight = std::int32_t;
using UnsignedEdgeWeight = std::uint32_t;
#endif // KAMINPAR_64BIT_WEIGHTS

using BlockID = std::uint32_t;
using BlockWeight = NodeWeight;

constexpr BlockID kInvalidBlockID = std::numeric_limits<BlockID>::max();
constexpr NodeID kInvalidNodeID = std::numeric_limits<NodeID>::max();
constexpr EdgeID kInvalidEdgeID = std::numeric_limits<EdgeID>::max();
constexpr NodeWeight kInvalidNodeWeight = std::numeric_limits<NodeWeight>::max();
constexpr EdgeWeight kInvalidEdgeWeight = std::numeric_limits<EdgeWeight>::max();
constexpr BlockWeight kInvalidBlockWeight = std::numeric_limits<BlockWeight>::max();

} // namespace kaminpar::shm
#endif // __cplusplus

// C interface:
#ifdef KAMINPAR_64BIT_NODE_IDS
typedef uint64_t kaminpar_node_id_t;
#else  // KAMINPAR_64BIT_NODE_IDS
typedef uint32_t kaminpar_node_id_t;
#endif // KAMINPAR_64BIT_NODE_IDS

#ifdef KAMINPAR_64BIT_EDGE_IDS
typedef uint64_t kaminpar_edge_id_t;
#else  // KAMINPAR_64BIT_EDGE_IDS
typedef uint32_t kaminpar_edge_id_t;
#endif // KAMINPAR_64BIT_EDGE_IDS

#ifdef KAMINPAR_64BIT_WEIGHTS
typedef int64_t kaminpar_node_weight_t;
typedef int64_t kaminpar_edge_weight_t;
typedef uint64_t kaminpar_unsigned_edge_weight_t;
typedef uint64_t kaminpar_unsigned_node_weight_t;
#else  // KAMINPAR_64BIT_WEIGHTS
typedef int32_t kaminpar_node_weight_t;
typedef uint32_t kaminpar_unsigned_node_weight_t;
typedef int32_t kaminpar_edge_weight_t;
typedef uint32_t kaminpar_unsigned_edge_weight_t;
#endif // KAMINPAR_64BIT_WEIGHTS

typedef uint32_t kaminpar_block_id_t;
typedef kaminpar_node_weight_t kaminpar_block_weight_t;

#ifdef __cplusplus
namespace kaminpar::shm {

enum class NodeOrdering {
  NATURAL,
  DEGREE_BUCKETS,
  EXTERNAL_DEGREE_BUCKETS,
  IMPLICIT_DEGREE_BUCKETS
};

enum class EdgeOrdering {
  NATURAL,
  COMPRESSION
};

//
// Coarsening
//

enum class CoarseningAlgorithm {
  NOOP,
  CLUSTERING,
  OVERLAY_CLUSTERING,
  SPARSIFYING_CLUSTERING,
};

enum class SparsificationAlgorithm {
  UNIFORM_RANDOM_SAMPLING,
  WEIGHTED_FOREST_FIRE,
  K_NEIGHBOUR,
  K_NEIGHBOUR_SPANNING_TREE,
  WEIGHT_THRESHOLD,
  RANDOM_WITH_REPLACEMENT,
  RANDOM_WITHOUT_REPLACEMENT,
  INDEPENDENT_RANDOM,
  THRESHOLD,
  UNBIASED_THRESHOLD
};

enum class ScoreFunctionSection {
  WEIGHT,
  WEIGHTED_FOREST_FIRE,
  FOREST_FIRE,
};

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

enum class LabelPropagationImplementation {
  SINGLE_PHASE,
  TWO_PHASE,
  GROWING_HASH_TABLES
};

enum class TwoHopStrategy {
  DISABLE,
  MATCH,
  MATCH_THREADWISE,
  CLUSTER,
  CLUSTER_THREADWISE,
};

enum class IsolatedNodesClusteringStrategy {
  KEEP,
  MATCH,
  CLUSTER,
  MATCH_DURING_TWO_HOP,
  CLUSTER_DURING_TWO_HOP,
};

enum class TieBreakingStrategy {
  GEOMETRIC,
  UNIFORM,
};

enum class ContractionAlgorithm {
  BUFFERED,
  UNBUFFERED,
  UNBUFFERED_NAIVE,
};

enum class ContractionImplementation {
  SINGLE_PHASE,
  TWO_PHASE,
  GROWING_HASH_TABLES
};

struct LabelPropagationCoarseningContext {
  std::size_t num_iterations;
  NodeID large_degree_threshold;
  NodeID max_num_neighbors;

  LabelPropagationImplementation impl;
  bool relabel_before_second_phase;

  TwoHopStrategy two_hop_strategy;
  double two_hop_threshold;

  IsolatedNodesClusteringStrategy isolated_nodes_strategy;

  TieBreakingStrategy tie_breaking_strategy;
};

struct ContractionCoarseningContext {
  ContractionAlgorithm algorithm;
  ContractionImplementation unbuffered_implementation;

  double edge_buffer_fill_fraction;
};

struct ClusterCoarseningContext {
  ClusteringAlgorithm algorithm;
  LabelPropagationCoarseningContext lp;

  ClusterWeightLimit cluster_weight_limit;
  double cluster_weight_multiplier;

  double shrink_factor;

  std::size_t max_mem_free_coarsening_level;

  bool forced_kc_level;
  bool forced_pc_level;
  double forced_level_upper_factor;
  double forced_level_lower_factor;
};

struct OverlayClusterCoarseningContext {
  int num_levels;
  int max_level;
};

struct CoarseningContext {
  CoarseningAlgorithm algorithm;

  ClusterCoarseningContext clustering;
  OverlayClusterCoarseningContext overlay_clustering;
  ContractionCoarseningContext contraction;

  NodeID contraction_limit;

  double convergence_threshold;
};

struct SparsificationContext {
  SparsificationAlgorithm algorithm;
  ScoreFunctionSection score_function;

  float density_target_factor;
  float reduction_target_factor;
  float laziness_factor;

  bool no_approx;

  float wff_target_burnt_ratio;
  float wff_pf;
};

//
// Refinement
//

enum class RefinementAlgorithm {
  LABEL_PROPAGATION,
  KWAY_FM,
  GREEDY_BALANCER,
  JET,
  MTKAHYPAR,
  NOOP,
};

enum class FMStoppingRule {
  SIMPLE,
  ADAPTIVE,
};

enum class GainCacheStrategy {
  COMPACT_HASHING,
  COMPACT_HASHING_LARGE_K,
  SPARSE,
  SPARSE_LARGE_K,
  HASHING,
  HASHING_LARGE_K,
  DENSE,
  DENSE_LARGE_K,
  ON_THE_FLY,
};

struct LabelPropagationRefinementContext {
  std::size_t num_iterations;
  NodeID large_degree_threshold;
  NodeID max_num_neighbors;

  LabelPropagationImplementation impl;

  TieBreakingStrategy tie_breaking_strategy;
};

struct KwayFMRefinementContext {
  NodeID num_seed_nodes;
  double alpha;
  int num_iterations;
  bool unlock_locally_moved_nodes;
  bool unlock_seed_nodes;
  bool use_exact_abortion_threshold;
  double abortion_threshold;

  GainCacheStrategy gain_cache_strategy;
  EdgeID constant_high_degree_threshold;
  double k_based_high_degree_threshold;

  int minimal_parallelism;

  bool dbg_compute_batch_stats;
  bool dbg_report_progress;
};

struct JetRefinementContext {
  int num_iterations;
  int num_fruitless_iterations;
  double fruitless_threshold;
  int num_rounds_on_fine_level;
  int num_rounds_on_coarse_level;
  double initial_gain_temp_on_fine_level;
  double final_gain_temp_on_fine_level;
  double initial_gain_temp_on_coarse_level;
  double final_gain_temp_on_coarse_level;
  RefinementAlgorithm balancing_algorithm;
};

struct MtKaHyParRefinementContext {
  std::string config_filename;
  std::string coarse_config_filename;
  std::string fine_config_filename;
};

struct BalancerRefinementContext {};

struct RefinementContext {
  std::vector<RefinementAlgorithm> algorithms;
  LabelPropagationRefinementContext lp;
  KwayFMRefinementContext kway_fm;
  BalancerRefinementContext balancer;
  JetRefinementContext jet;
  MtKaHyParRefinementContext mtkahypar;

  [[nodiscard]] bool includes_algorithm(const RefinementAlgorithm algorithm) const {
    return std::find(algorithms.begin(), algorithms.end(), algorithm) != algorithms.end();
  }
};

//
// Initial Partitioning
//

enum class InitialPartitioningMode {
  SEQUENTIAL,
  ASYNCHRONOUS_PARALLEL,
  SYNCHRONOUS_PARALLEL,
  COMMUNITIES,
};

struct InitialCoarseningContext {
  NodeID contraction_limit;
  double convergence_threshold;
  NodeID large_degree_threshold;

  ClusterWeightLimit cluster_weight_limit;
  double cluster_weight_multiplier;
};

struct InitialRefinementContext {
  bool disabled;

  FMStoppingRule stopping_rule;
  NodeID num_fruitless_moves;
  double alpha;

  std::size_t num_iterations;
  double improvement_abortion_threshold;
};

struct InitialPoolPartitionerContext {
  InitialRefinementContext refinement;

  double repetition_multiplier;

  int min_num_repetitions;
  int min_num_non_adaptive_repetitions;
  int max_num_repetitions;
  int num_seed_iterations;

  bool use_adaptive_bipartitioner_selection;

  bool enable_bfs_bipartitioner;
  bool enable_ggg_bipartitioner;
  bool enable_random_bipartitioner;
};

struct InitialPartitioningContext {
  InitialCoarseningContext coarsening;
  InitialPoolPartitionerContext pool;
  InitialRefinementContext refinement;

  bool refine_pool_partition;
  bool use_adaptive_epsilon;
};

//
// Application level
//

struct PartitionContext {
  NodeID original_n = kInvalidNodeID;
  NodeID n = kInvalidNodeID;
  EdgeID m = kInvalidEdgeID;
  NodeWeight original_total_node_weight = kInvalidNodeWeight;
  NodeWeight total_node_weight = kInvalidNodeWeight;
  EdgeWeight total_edge_weight = kInvalidEdgeWeight;
  NodeWeight max_node_weight = kInvalidNodeWeight;

  BlockID k;

  [[nodiscard]] BlockWeight perfectly_balanced_block_weight(const BlockID block) const {
    return std::ceil(1.0 * _unrelaxed_max_block_weights[block] / (1 + inferred_epsilon()));
  }

  [[nodiscard]] BlockWeight max_block_weight(const BlockID block) const {
    return _max_block_weights[block];
  }

  [[nodiscard]] BlockWeight total_max_block_weights(const BlockID begin, const BlockID end) const {
    if (_uniform_block_weights) {
      return _max_block_weights[begin] * (end - begin);
    }

    return std::accumulate(
        _max_block_weights.begin() + begin,
        _max_block_weights.begin() + end,
        static_cast<BlockWeight>(0)
    );
  }

  [[nodiscard]] BlockWeight
  total_unrelaxed_max_block_weights(const BlockID begin, const BlockID end) const {
    if (_uniform_block_weights) {
      return (1.0 + inferred_epsilon()) * std::ceil(1.0 * (end - begin) * total_node_weight / k);
    }

    return std::accumulate(
        _unrelaxed_max_block_weights.begin() + begin,
        _unrelaxed_max_block_weights.begin() + end,
        static_cast<BlockWeight>(0)
    );
  }

  [[nodiscard]] double epsilon() const {
    return _epsilon < 0.0 ? inferred_epsilon() : _epsilon;
  }

  [[nodiscard]] double infer_epsilon(const NodeWeight actual_total_node_weight) const {
    if (_uniform_block_weights) {
      const double max = (1.0 + _epsilon) * std::ceil(1.0 * original_total_node_weight / k);
      return max / std::ceil(1.0 * actual_total_node_weight / k) - 1.0;
    }

    return 1.0 * _total_max_block_weights / actual_total_node_weight - 1.0;
  }

  [[nodiscard]] double inferred_epsilon() const {
    return infer_epsilon(total_node_weight);
  }

  void set_epsilon(const double eps) {
    _epsilon = eps;
  }

  [[nodiscard]] bool has_epsilon() const {
    return _epsilon > 0.0;
  }

  [[nodiscard]] bool has_uniform_block_weights() const {
    return _uniform_block_weights;
  }

  void setup(
      const class AbstractGraph &graph,
      BlockID k,
      double epsilon,
      bool relax_max_block_weights = false
  );

  void setup(
      const class AbstractGraph &graph,
      std::vector<BlockWeight> max_block_weights,
      bool relax_max_block_weights = false
  );

private:
  std::vector<BlockWeight> _max_block_weights{};
  std::vector<BlockWeight> _unrelaxed_max_block_weights{};

  BlockWeight _total_max_block_weights = 0;
  double _epsilon = -1.0;
  bool _uniform_block_weights = false;
};

struct ParallelContext {
  int num_threads;
};

struct DebugContext {
  std::string graph_name;

  // Options for dumping coarse graphs and intermediate partitions
  std::string dump_graph_filename;
  std::string dump_partition_filename;
  bool dump_toplevel_graph;
  bool dump_toplevel_partition;
  bool dump_coarsest_graph;
  bool dump_coarsest_partition;
  bool dump_graph_hierarchy;
  bool dump_partition_hierarchy;
};

enum class PartitioningMode {
  DEEP,
  VCYCLE,
  RB,
  KWAY,
};

struct PartitioningContext {
  PartitioningMode mode;

  InitialPartitioningMode deep_initial_partitioning_mode;
  double deep_initial_partitioning_load;
  int min_consecutive_seq_bipartitioning_levels;
  bool refine_after_extending_partition;

  bool use_lazy_subgraph_memory;

  std::vector<BlockID> vcycles;
  bool restrict_vcycle_refinement;

  bool rb_enable_kway_toplevel_refinement;
};

struct GraphCompressionContext {
  bool enabled;

  bool high_degree_encoding = false;
  NodeID high_degree_threshold = kInvalidNodeID;
  NodeID high_degree_part_length = kInvalidNodeID;
  bool interval_encoding = false;
  NodeID interval_length_treshold = kInvalidNodeID;
  bool streamvbyte_encoding = false;

  double compression_ratio = -1;
  std::int64_t size_reduction = -1;
  std::size_t num_high_degree_nodes = std::numeric_limits<std::size_t>::max();
  std::size_t num_high_degree_parts = std::numeric_limits<std::size_t>::max();
  std::size_t num_interval_nodes = std::numeric_limits<std::size_t>::max();
  std::size_t num_intervals = std::numeric_limits<std::size_t>::max();

  void setup(const class Graph &graph);
};

struct Context {
  GraphCompressionContext compression;
  NodeOrdering node_ordering;
  EdgeOrdering edge_ordering;

  PartitioningContext partitioning;
  PartitionContext partition;
  CoarseningContext coarsening;
  SparsificationContext sparsification;
  InitialPartitioningContext initial_partitioning;
  RefinementContext refinement;
  ParallelContext parallel;
  DebugContext debug;
};

} // namespace kaminpar::shm
#endif // __cplusplus

//
// Configuration presets
//

#ifdef __cplusplus
namespace kaminpar::shm {

std::unordered_set<std::string> get_preset_names();

Context create_context_by_preset_name(const std::string &name);

Context create_default_context();
Context create_fast_context();
Context create_strong_context();

Context create_terapart_context();
Context create_terapart_strong_context();
Context create_terapart_largek_context();

Context create_largek_context();
Context create_largek_fast_context();
Context create_largek_strong_context();

Context create_jet_context(int rounds = 1);
Context create_noref_context();

Context create_vcycle_context(bool restrict_refinement = false);

Context create_esa21_smallk_context();
Context create_esa21_largek_context();
Context create_esa21_largek_fast_context();
Context create_esa21_strong_context();

} // namespace kaminpar::shm
#endif // __cplusplus

// C interface
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct kaminpar_context_t kaminpar_context_t;
kaminpar_context_t *kaminpar_create_context_by_preset_name(const char *name);
kaminpar_context_t *kaminpar_create_default_context();
kaminpar_context_t *kaminpar_create_strong_context();
kaminpar_context_t *kaminpar_create_terapart_context();
kaminpar_context_t *kaminpar_create_largek_context();
kaminpar_context_t *kaminpar_create_vcycle_context(bool restrict_refinement);
void kaminpar_context_free(kaminpar_context_t *ctx);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

//
// Shared-memory partitioner interface
//

#ifdef __cplusplus
namespace kaminpar {

namespace shm {

class Graph;

} // namespace shm

class KaMinPar {
public:
  KaMinPar(int num_threads, shm::Context ctx);

  KaMinPar(const KaMinPar &) = delete;
  KaMinPar &operator=(const KaMinPar &) = delete;

  KaMinPar(KaMinPar &&) noexcept = default;
  KaMinPar &operator=(KaMinPar &&) noexcept = default;

  ~KaMinPar();

  static void reseed(int seed);

  /*!
   * Sets the verbosity of the partitioner.
   *
   * @param output_level Integer verbosity level, higher values mean more output.
   */
  void set_output_level(OutputLevel output_level);

  /*!
   * Sets the maximum depth of the timer tree. Only meaningful if the output level is set to
   * `APPLICATION` or `EXPERIMENT`.
   *
   * @param max_timer_depth The maximum depth of the timer stack.
   */
  void set_max_timer_depth(int max_timer_depth);

  /*!
   * Returns a non-const reference to the context object, which can be used to configure the
   * partitioning process.
   *
   * @return Reference to the context object.
   */
  shm::Context &context();

  /*!
   * Sets the graph to be partitioned by taking temporary ownership of the given pointers. In
   * particular, the partitioner might modify the data pointed to. The caller is responsible for
   * free'ing the memory.
   *
   * @param xadj Array of length `n + 1`, where `xadj[u]` points to the first neighbor of node `u`
   * in `adjncy`. In other words, the neighbors of `u` are `adjncy[xadj[u]..xadj[u+1]-1]`.
   * @param adjncy Array of length `xadj[n]` storing the neighbors of all nodes.
   * @param vwgt Array of length `n` storing the weight of each node. If the nodes are unweighted,
   * pass `nullptr`.
   * @param adjwgt Array of length `xadj[n]` storing the weight of each edge. Note that reverse
   * edges must be assigned the same weight. If the edges are unweighted, pass `nullptr`.
   */
  void borrow_and_mutate_graph(
      std::span<shm::EdgeID> xadj,
      std::span<shm::NodeID> adjncy,
      std::span<shm::NodeWeight> vwgt = {},
      std::span<shm::EdgeWeight> adjwgt = {}
  );

  /*!
   * Sets the graph to be partitioned by copying the data pointed to by the given pointers.
   *
   * @param xadj Array of length `n + 1`, where `xadj[u]` points to the first neighbor of node `u`
   * in `adjncy`. In other words, the neighbors of `u` are `adjncy[xadj[u]..xadj[u+1]-1]`.
   * @param adjncy Array of length `xadj[n]` storing the neighbors of all nodes.
   * @param vwgt Array of length `n` storing the weight of each node. If the nodes are unweighted,
   * pass `nullptr`.
   * @param adjwgt Array of length `xadj[n]` storing the weight of each edge. Note that reverse
   * edges must be assigned the same weight. If the edges are unweighted, pass `nullptr`.
   */
  void copy_graph(
      std::span<const shm::EdgeID> xadj,
      std::span<const shm::NodeID> adjncy,
      std::span<const shm::NodeWeight> vwgt = {},
      std::span<const shm::EdgeWeight> adjwgt = {}
  );

  /*!
   * Sets the graph to be partitioned.
   *
   * @param graph The graph to be partitioned.
   */
  void set_graph(shm::Graph graph);

  /*!
   * Partitions the graph set by `borrow_and_mutate_graph()` or `copy_graph()` into `k` blocks with
   * a maximum imbalance of 3%.
   *
   * @param k Number of blocks.
   * @param[out] partition Span of length `n` to store the partitioning.
   *
   * @return Expected edge cut of the partition.
   */
  shm::EdgeWeight compute_partition(shm::BlockID k, std::span<shm::BlockID> partition);

  /*!
   * Partitions the graph set by `borrow_and_mutate_graph()` or `copy_graph()` into `k` blocks with
   * a maximum imbalance of `epsilon`.
   *
   * @param k Number of blocks.
   * @param epsilon Balance constraint (e.g., 0.03 for max 3% imbalance).
   * @param[out] partition Span of length `n` to store the partitioning.
   *
   * @return Expected edge cut of the partition.
   */
  shm::EdgeWeight
  compute_partition(shm::BlockID k, double epsilon, std::span<shm::BlockID> partition);

  /*!
   * Partitions the graph set by `borrow_and_mutate_graph()` or `copy_graph()` such that the
   * weight of each block is upper bounded by `max_block_weights`. The number of blocks is given
   * implicitly by the size of the vector.
   *
   * @param max_block_weights Maximum weight for each block of the partition.
   * @param[out] partition Span of length `n` to store the partitioning.
   *
   * @return Expected edge cut of the partition.
   */
  shm::EdgeWeight compute_partition(
      std::vector<shm::BlockWeight> max_block_weights, std::span<shm::BlockID> partition
  );

  /*!
   * Partitions the graph set by `borrow_and_mutate_graph()` or `copy_graph()` such that the
   * weight of each block is upper bounded by `max_block_weight_factors` times the total node weigh
   * of the graph. The number of blocks is given implicitly by the size of the vector.
   *
   * @param max_block_weight_factors Maximum weight factor for each block of the partition.
   * @param[out] partition Span of length `n` to store the partitioning.
   *
   * @return Expected edge cut of the partition.
   */
  shm::EdgeWeight compute_partition(
      std::vector<double> max_block_weight_factors, std::span<shm::BlockID> partition
  );

  const shm::Graph *graph();

private:
  shm::EdgeWeight compute_partition(std::span<shm::BlockID> partition);

  int _num_threads;

  int _max_timer_depth = std::numeric_limits<int>::max();
  OutputLevel _output_level = OutputLevel::APPLICATION;
  shm::Context _ctx;

  std::unique_ptr<shm::Graph> _graph_ptr;
  tbb::global_control _gc;

  bool _was_rearranged = false;
};

} // namespace kaminpar
#endif // __cplusplus

// C interface
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct kaminpar_t kaminpar_t;

kaminpar_t *kaminpar_create(int num_threads, kaminpar_context_t *ctx);
void kaminpar_free(kaminpar_t *kaminpar);

void kaminpar_set_output_level(kaminpar_t *kaminpar, kaminpar_output_level_t output_level);
void kaminpar_set_max_timer_depth(kaminpar_t *kaminpar, int max_timer_depth);

void kaminpar_copy_graph(
    kaminpar_t *kaminpar,
    kaminpar_node_id_t n,
    const kaminpar_edge_id_t *xadj,
    const kaminpar_node_id_t *adjncy,
    const kaminpar_node_weight_t *vwgt,
    const kaminpar_edge_weight_t *adjwgt
);

void kaminpar_borrow_and_mutate_graph(
    kaminpar_t *kaminpar,
    kaminpar_node_id_t n,
    kaminpar_edge_id_t *xadj,
    kaminpar_node_id_t *adjncy,
    kaminpar_node_weight_t *vwgt,
    kaminpar_edge_weight_t *adjwgt
);

kaminpar_edge_weight_t kaminpar_compute_partition(
    kaminpar_t *kaminpar, kaminpar_block_id_t k, kaminpar_block_id_t *partition
);

kaminpar_edge_weight_t kaminpar_compute_partition_with_epsilon(
    kaminpar_t *kaminpar, kaminpar_block_id_t k, double epsilon, kaminpar_block_id_t *partition
);

kaminpar_edge_weight_t kaminpar_compute_partition_with_max_block_weight_factors(
    kaminpar_t *kaminpar,
    kaminpar_block_id_t k,
    const double *max_block_weight_factors,
    kaminpar_block_id_t *partition
);

kaminpar_edge_weight_t kaminpar_compute_partition_with_max_block_weights(
    kaminpar_t *kaminpar,
    kaminpar_block_id_t k,
    const kaminpar_block_weight_t *max_block_weights,
    kaminpar_block_id_t *partition
);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
