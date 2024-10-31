/*******************************************************************************
 * Public library interface of KaMinPar.
 *
 * @file:   kaminpar.h
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 ******************************************************************************/
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <tbb/global_control.h>

#define KAMINPAR_VERSION_MAJOR 3
#define KAMINPAR_VERSION_MINOR 0
#define KAMINPAR_VERSION_PATCH 0

namespace kaminpar {

enum class OutputLevel : std::uint8_t {
  QUIET,       //! Disable all output to stdout.
  PROGRESS,    //! Continuously output progress information while partitioning.
  APPLICATION, //! Also output the application banner and context summary.
  EXPERIMENT,  //! Also output information only relevant for benchmarking.
  DEBUG,       //! Also output (a sane amount) of debug information.
};

} // namespace kaminpar

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
using EdgeWeight = std::int32_t;
using UnsignedEdgeWeight = std::uint32_t;
using UnsignedNodeWeight = std::uint32_t;
#endif // KAMINPAR_64BIT_WEIGHTS

using BlockID = std::uint32_t;
using BlockWeight = NodeWeight;

constexpr BlockID kInvalidBlockID = std::numeric_limits<BlockID>::max();
constexpr NodeID kInvalidNodeID = std::numeric_limits<NodeID>::max();
constexpr EdgeID kInvalidEdgeID = std::numeric_limits<EdgeID>::max();
constexpr NodeWeight kInvalidNodeWeight = std::numeric_limits<NodeWeight>::max();
constexpr EdgeWeight kInvalidEdgeWeight = std::numeric_limits<EdgeWeight>::max();
constexpr BlockWeight kInvalidBlockWeight = std::numeric_limits<BlockWeight>::max();

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

enum class SecondPhaseSelectionStrategy {
  HIGH_DEGREE,
  FULL_RATING_MAP
};

enum class SecondPhaseAggregationStrategy {
  NONE,
  DIRECT,
  BUFFERED
};

enum class TwoHopStrategy {
  DISABLE,
  MATCH,
  MATCH_THREADWISE,
  CLUSTER,
  CLUSTER_THREADWISE,
  LEGACY,
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
  BUFFERED_LEGACY,
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

  SecondPhaseSelectionStrategy second_phase_selection_strategy;
  SecondPhaseAggregationStrategy second_phase_aggregation_strategy;
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

struct CoarseningContext {
  CoarseningAlgorithm algorithm;

  ClusterCoarseningContext clustering;
  ContractionCoarseningContext contraction;

  NodeID contraction_limit;

  double convergence_threshold;
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

  SecondPhaseSelectionStrategy second_phase_selection_strategy;
  SecondPhaseAggregationStrategy second_phase_aggregation_strategy;
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
};

//
// Application level
//

class AbstractGraph;
class Graph;
struct PartitionContext;

struct BlockWeightsContext {
  void setup(const PartitionContext &ctx, const bool parallel = true);
  void setup(const PartitionContext &ctx, const BlockID input_k, const bool parallel = true);

  [[nodiscard]] BlockWeight max(BlockID b) const {
    return _max_block_weights[b];
  }

  [[nodiscard]] const std::vector<BlockWeight> &all_max() const;

  [[nodiscard]] BlockWeight perfectly_balanced(BlockID b) const {
    return _perfectly_balanced_block_weights[b];
  }

  [[nodiscard]] const std::vector<BlockWeight> &all_perfectly_balanced() const;

private:
  std::vector<BlockWeight> _perfectly_balanced_block_weights;
  std::vector<BlockWeight> _max_block_weights;
};

struct PartitionContext {
  double epsilon;
  BlockID k;

  BlockWeightsContext block_weights{};
  void setup_block_weights();

  NodeID n = kInvalidNodeID;
  EdgeID m = kInvalidEdgeID;
  NodeWeight total_node_weight = kInvalidNodeWeight;
  EdgeWeight total_edge_weight = kInvalidEdgeWeight;
  NodeWeight max_node_weight = kInvalidNodeWeight;

  void setup(const AbstractGraph &graph, const bool setup_block_weights = true);
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
};

struct GraphCompressionContext {
  bool enabled;

  bool compressed_edge_weights = false;
  bool high_degree_encoding = false;
  NodeID high_degree_threshold = kInvalidNodeID;
  NodeID high_degree_part_length = kInvalidNodeID;
  bool interval_encoding = false;
  NodeID interval_length_treshold = kInvalidNodeID;
  bool run_length_encoding = false;
  bool streamvbyte_encoding = false;

  double compression_ratio = -1;
  std::int64_t size_reduction = -1;
  std::size_t num_high_degree_nodes = std::numeric_limits<std::size_t>::max();
  std::size_t num_high_degree_parts = std::numeric_limits<std::size_t>::max();
  std::size_t num_interval_nodes = std::numeric_limits<std::size_t>::max();
  std::size_t num_intervals = std::numeric_limits<std::size_t>::max();

  void setup(const Graph &graph);
};

struct Context {
  GraphCompressionContext compression;
  NodeOrdering node_ordering;
  EdgeOrdering edge_ordering;

  PartitioningContext partitioning;
  PartitionContext partition;
  CoarseningContext coarsening;
  InitialPartitioningContext initial_partitioning;
  RefinementContext refinement;
  ParallelContext parallel;
  DebugContext debug;

  void setup(const Graph &graph);
};
} // namespace kaminpar::shm

//
// Configuration presets
//

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

} // namespace kaminpar::shm

//
// Shared-memory partitioner interface
//

namespace kaminpar {

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
   * @param n The number of nodes in the graph.
   * @param xadj Array of length `n + 1`, where `xadj[u]` points to the first neighbor of node `u`
   * in `adjncy`. In other words, the neighbors of `u` are `adjncy[xadj[u]..xadj[u+1]-1]`.
   * @param adjncy Array of length `xadj[n]` storing the neighbors of all nodes.
   * @param vwgt Array of length `n` storing the weight of each node. If the nodes are unweighted,
   * pass `nullptr`.
   * @param adjwgt Array of length `xadj[n]` storing the weight of each edge. Note that reverse
   * edges must be assigned the same weight. If the edges are unweighted, pass `nullptr`.
   */
  void borrow_and_mutate_graph(
      shm::NodeID n,
      shm::EdgeID *xadj,
      shm::NodeID *adjncy,
      shm::NodeWeight *vwgt,
      shm::EdgeWeight *adjwgt
  );

  /*!
   * Sets the graph to be partitioned by copying the data pointed to by the given pointers.
   *
   * @param n The number of nodes in the graph.
   * @param xadj Array of length `n + 1`, where `xadj[u]` points to the first neighbor of node `u`
   * in `adjncy`. In other words, the neighbors of `u` are `adjncy[xadj[u]..xadj[u+1]-1]`.
   * @param adjncy Array of length `xadj[n]` storing the neighbors of all nodes.
   * @param vwgt Array of length `n` storing the weight of each node. If the nodes are unweighted,
   * pass `nullptr`.
   * @param adjwgt Array of length `xadj[n]` storing the weight of each edge. Note that reverse
   * edges must be assigned the same weight. If the edges are unweighted, pass `nullptr`.
   */
  void copy_graph(
      shm::NodeID n,
      const shm::EdgeID *const xadj,
      const shm::NodeID *const adjncy,
      const shm::NodeWeight *const vwgt,
      const shm::EdgeWeight *const adjwgt
  );

  /*!
   * Sets the graph to be partitioned.
   *
   * @param graph The graph to be partitioned.
   */
  void set_graph(shm::Graph graph);

  /*!
   * Partitions the graph set by `borrow_and_mutate_graph()` or `copy_graph()` into `k` blocks.
   *
   * @param k The number of blocks to partition the graph into.
   * @param partition Array of length `n` for storing the partition. The caller is reponsible for
   * allocating and freeing the memory.
   *
   * @return The edge-cut of the partition.
   */
  shm::EdgeWeight
  compute_partition(shm::BlockID k, shm::BlockID *partition, bool use_initial_node_ordering = true);

  const shm::Graph *graph();

private:
  int _num_threads;

  int _max_timer_depth = std::numeric_limits<int>::max();
  OutputLevel _output_level = OutputLevel::APPLICATION;
  shm::Context _ctx;

  std::unique_ptr<shm::Graph> _graph_ptr;
  tbb::global_control _gc;

  bool _was_rearranged = false;
};

} // namespace kaminpar
