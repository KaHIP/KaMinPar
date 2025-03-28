/*******************************************************************************
 * Public interface of the distributed partitioner.
 *
 * @file:   dkaminpar.h
 * @author: Daniel Seemaier
 * @date:   30.01.2023
 ******************************************************************************/
#ifndef DKAMINPAR_H
#define DKAMINPAR_H

#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <unordered_set>

#include <mpi.h>
#include <tbb/global_control.h>

#include <kaminpar.h>

namespace kaminpar::mpi {

using PEID = int;
using UPEID = unsigned int;

} // namespace kaminpar::mpi

namespace kaminpar::dist {

using GlobalNodeID = std::uint64_t;
using GlobalNodeWeight = std::int64_t;
using GlobalEdgeID = std::uint64_t;
using GlobalEdgeWeight = std::int64_t;
using BlockWeight = std::int64_t;

using mpi::PEID;
using mpi::UPEID;

using shm::BlockID;
using shm::EdgeID;
using shm::NodeID;

#ifdef KAMINPAR_64BIT_LOCAL_WEIGHTS

using NodeWeight = std::int64_t;
using EdgeWeight = std::int64_t;
using UnsignedEdgeWeight = std::uint64_t;

#else // KAMINPAR_64BIT_LOCAL_WEIGHTS

using NodeWeight = std::int32_t;
using EdgeWeight = std::int32_t;
using UnsignedEdgeWeight = std::uint32_t;

#endif

constexpr NodeID kInvalidNodeID = std::numeric_limits<NodeID>::max();
constexpr GlobalNodeID kInvalidGlobalNodeID = std::numeric_limits<GlobalNodeID>::max();
constexpr NodeWeight kInvalidNodeWeight = std::numeric_limits<NodeWeight>::max();
constexpr GlobalNodeWeight kInvalidGlobalNodeWeight = std::numeric_limits<GlobalNodeWeight>::max();
constexpr EdgeID kInvalidEdgeID = std::numeric_limits<EdgeID>::max();
constexpr GlobalEdgeID kInvalidGlobalEdgeID = std::numeric_limits<GlobalEdgeID>::max();
constexpr EdgeWeight kInvalidEdgeWeight = std::numeric_limits<EdgeWeight>::max();
constexpr GlobalEdgeWeight kInvalidGlobalEdgeWeight = std::numeric_limits<GlobalEdgeWeight>::max();
constexpr BlockID kInvalidBlockID = std::numeric_limits<BlockID>::max();
constexpr BlockWeight kInvalidBlockWeight = std::numeric_limits<BlockWeight>::max();

} // namespace kaminpar::dist

namespace kaminpar::dist {

enum class PartitioningMode {
  KWAY,
  DEEP,
};

enum class ClusteringAlgorithm {
  GLOBAL_NOOP,
  GLOBAL_LP,
  GLOBAL_HEM,
  GLOBAL_HEM_LP,

  LOCAL_NOOP,
  LOCAL_LP,
};

enum class InitialPartitioningAlgorithm {
  KAMINPAR,
  MTKAHYPAR,
  RANDOM,
};

enum class RefinementAlgorithm {
  NOOP,
  BATCHED_LP_REFINER,
  COLORED_LP_REFINER,
  JET_REFINER,
  HYBRID_NODE_BALANCER,
  HYBRID_CLUSTER_BALANCER,
  MTKAHYPAR_REFINER,
};

enum class GraphOrdering {
  NATURAL,
  DEGREE_BUCKETS,
  COLORING,
};

enum class GraphDistribution {
  BALANCED_NODES,
  BALANCED_EDGES,
  BALANCED_MEMORY_SPACE
};

enum class LabelPropagationMoveExecutionStrategy {
  PROBABILISTIC,
  BEST_MOVES,
  LOCAL_MOVES,
};

enum class ClusterSizeStrategy {
  ZERO,
  ONE,
  MAX_OVERLOAD,
  AVG_OVERLOAD,
  MIN_OVERLOAD,
};

enum class ClusterStrategy {
  SINGLETONS,
  LP,
  GREEDY_BATCH_PREFIX,
};

enum class GainCacheStrategy {
  ON_THE_FLY,
  COMPACT_HASHING,
  LAZY_COMPACT_HASHING,
};

enum class ActiveSetStrategy {
  NONE,
  LOCAL,
  GLOBAL,
};

struct ParallelContext {
  std::size_t num_threads;
  std::size_t num_mpis;
};

struct ChunksContext {
  int total_num_chunks;
  int fixed_num_chunks;
  int min_num_chunks;
  bool scale_chunks_with_threads;

  [[nodiscard]] int compute(const ParallelContext &parallel) const;
};

struct LabelPropagationCoarseningContext {
  int num_iterations;
  NodeID passive_high_degree_threshold;
  NodeID active_high_degree_threshold;
  NodeID max_num_neighbors;
  bool merge_singleton_clusters;
  double merge_nonadjacent_clusters_threshold;

  ChunksContext chunks;

  bool ignore_ghost_nodes;
  bool keep_ghost_clusters;

  bool sync_cluster_weights;
  bool enforce_cluster_weights;

  bool prevent_cyclic_moves;

  ActiveSetStrategy active_set_strategy;

  [[nodiscard]] bool should_merge_nonadjacent_clusters(NodeID old_n, NodeID new_n) const;
};

struct HEMCoarseningContext {
  ChunksContext chunks;

  double small_color_blacklist;
  bool only_blacklist_input_level;
  bool ignore_weight_limit;
};

struct ColoredLabelPropagationRefinementContext {
  int num_iterations;
  int num_move_execution_iterations;
  int num_probabilistic_move_attempts;
  bool sort_by_rel_gain;

  ChunksContext coloring_chunks;

  double small_color_blacklist;
  bool only_blacklist_input_level;

  bool track_local_block_weights;
  bool use_active_set;

  LabelPropagationMoveExecutionStrategy move_execution_strategy;
};

struct LabelPropagationRefinementContext {
  NodeID active_high_degree_threshold;
  int num_iterations;

  ChunksContext chunks;

  int num_move_attempts;
  bool ignore_probabilities;
};

struct MtKaHyParRefinementContext {
  std::string config_filename;
  std::string fine_config_filename;
  std::string coarse_config_filename;
  bool only_run_on_root;
};

struct CoarseningContext {
  // Global clustering
  std::size_t max_global_clustering_levels;
  ClusteringAlgorithm global_clustering_algorithm;
  LabelPropagationCoarseningContext global_lp;
  HEMCoarseningContext hem;

  // Local clustering
  std::size_t max_local_clustering_levels;
  ClusteringAlgorithm local_clustering_algorithm;
  LabelPropagationCoarseningContext local_lp;

  // Cluster weight limit
  NodeID contraction_limit;
  shm::ClusterWeightLimit cluster_weight_limit;
  double cluster_weight_multiplier;

  // Graph contraction
  double max_cnode_imbalance;
  bool migrate_cnode_prefix;
  bool force_perfect_cnode_balance;

  void setup(const ParallelContext &parallel);
};

struct InitialPartitioningContext {
  InitialPartitioningAlgorithm algorithm;
  shm::Context kaminpar;
};

struct NodeBalancerContext {
  int max_num_rounds;

  bool enable_sequential_balancing;
  NodeID seq_num_nodes_per_block;

  bool enable_parallel_balancing;
  double par_threshold;
  int par_num_dicing_attempts;
  bool par_accept_imbalanced_moves;
  bool par_enable_positive_gain_buckets;
  double par_gain_bucket_base;
  bool par_partial_buckets;
  bool par_update_pq_gains;
  int par_high_degree_update_interval;

  EdgeID par_high_degree_insertion_threshold;
  EdgeID par_high_degree_update_thresold;
};

struct ClusterBalancerContext {
  int max_num_rounds;

  bool enable_sequential_balancing;
  NodeID seq_num_nodes_per_block;
  bool seq_full_pq;

  bool enable_parallel_balancing;
  double parallel_threshold;
  int par_num_dicing_attempts;
  bool par_accept_imbalanced;

  bool par_use_positive_gain_buckets;
  double par_gain_bucket_factor;

  double par_initial_rebalance_fraction;
  double par_rebalance_fraction_increase;

  ClusterSizeStrategy cluster_size_strategy;
  double cluster_size_multiplier;

  ClusterStrategy cluster_strategy;
  int cluster_rebuild_interval;

  bool switch_to_sequential_after_stallmate;
  bool switch_to_singleton_after_stallmate;
};

struct JetRefinementContext {
  int num_coarse_rounds;
  int num_fine_rounds;

  int num_iterations;
  int num_fruitless_iterations;
  double fruitless_threshold;

  bool dynamic_negative_gain_factor;

  double coarse_negative_gain_factor;
  double fine_negative_gain_factor;

  double initial_negative_gain_factor;
  double final_negative_gain_factor;

  RefinementAlgorithm balancing_algorithm;

  GainCacheStrategy gain_cache_strategy;
};

struct RefinementContext {
  std::vector<RefinementAlgorithm> algorithms;
  bool refine_coarsest_level;

  LabelPropagationRefinementContext lp;
  ColoredLabelPropagationRefinementContext colored_lp;
  NodeBalancerContext node_balancer;
  ClusterBalancerContext cluster_balancer;

  JetRefinementContext jet;

  MtKaHyParRefinementContext mtkahypar;

  [[nodiscard]] bool includes_algorithm(RefinementAlgorithm algorithm) const;
};

struct GraphCompressionContext {
  bool enabled;

  // Graph compression statistics
  double avg_compression_ratio = 0.0;
  double min_compression_ratio = 0.0;
  double max_compression_ratio = 0.0;

  std::size_t largest_compressed_graph = 0;
  std::size_t largest_compressed_graph_prev_size = 0;

  std::size_t largest_uncompressed_graph = 0;
  std::size_t largest_uncompressed_graph_after_size = 0;

  std::vector<std::size_t> compressed_graph_sizes = {};
  std::vector<std::size_t> uncompressed_graph_sizes = {};
  std::vector<NodeID> num_nodes = {};
  std::vector<EdgeID> num_edges = {};

  /*!
   * Setups the graph compression statistics of this context.
   *
   * @param graph The compressed graph of this process.
   */
  void setup(const class DistributedCompressedGraph &graph);
};

struct PartitionContext {
  GlobalNodeID global_n = kInvalidGlobalNodeID;
  NodeID n = kInvalidNodeID;
  NodeID total_n = kInvalidNodeID;
  GlobalEdgeID global_m = kInvalidGlobalEdgeID;
  EdgeID m = kInvalidEdgeID;
  GlobalNodeWeight global_total_node_weight = kInvalidGlobalNodeWeight;
  NodeWeight total_node_weight = kInvalidNodeWeight;
  GlobalNodeWeight global_max_node_weight = kInvalidGlobalNodeWeight;
  GlobalEdgeWeight global_total_edge_weight = kInvalidGlobalEdgeWeight;
  EdgeWeight total_edge_weight = kInvalidEdgeWeight;

  BlockID k = kInvalidBlockID;

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
      return (1.0 + inferred_epsilon()) *
             std::ceil(1.0 * (end - begin) * global_total_node_weight / k);
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

  [[nodiscard]] double infer_epsilon() const {
    if (_uniform_block_weights) {
      return _epsilon;
    }

    return 1.0 * _total_max_block_weights / global_total_node_weight - 1.0;
  }

  [[nodiscard]] double inferred_epsilon() const {
    return infer_epsilon();
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
      const class AbstractDistributedGraph &graph,
      BlockID k,
      double epsilon,
      bool relax_max_block_weights = false
  );

  void setup(
      const class AbstractDistributedGraph &graph,
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

struct PartitioningContext {
  PartitioningMode mode;

  BlockID initial_k;
  BlockID extension_k;

  bool avoid_toplevel_bipartitioning;
  bool enable_pe_splitting;
  bool simulate_singlethread;
};

struct DebugContext {
  std::string graph_filename;
  bool save_coarsest_graph;
  bool save_coarsest_partition;
  bool print_compression_details;
};

struct Context {
  GraphOrdering rearrange_by;

  PartitioningContext partitioning;
  PartitionContext partition;

  ParallelContext parallel;
  GraphCompressionContext compression;
  CoarseningContext coarsening;
  InitialPartitioningContext initial_partitioning;
  RefinementContext refinement;
  DebugContext debug;
};

Context create_context_by_preset_name(const std::string &name);
Context create_default_context();
Context create_strong_context();
std::unordered_set<std::string> get_preset_names();
} // namespace kaminpar::dist

namespace kaminpar::dist {
class DistributedGraph;
}

namespace kaminpar {
class dKaMinPar {
public:
  dKaMinPar(MPI_Comm comm, int num_threads, dist::Context ctx);

  dKaMinPar(const dKaMinPar &) = delete;
  dKaMinPar &operator=(const dKaMinPar &) = delete;

  dKaMinPar(dKaMinPar &&) noexcept = default;
  dKaMinPar &operator=(dKaMinPar &&) noexcept = default;

  ~dKaMinPar();

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
  dist::Context &context();

  void copy_graph(
      std::span<const dist::GlobalNodeID> node_distribution,
      std::span<const dist::GlobalEdgeID> nodes,
      std::span<const dist::GlobalNodeID> edges,
      std::span<const dist::GlobalNodeWeight> node_weights = {},
      std::span<const dist::GlobalEdgeWeight> edge_weights = {}
  );

  void set_graph(dist::DistributedGraph graph);

  /*!
   * Partitions the graph set by `copy_graph()` or `set_graph()` into `k` blocks with
   * a maximum imbalance of 3%.
   *
   * @param k Number of blocks.
   * @param[out] partition Span of length `n` to store the partitioning.
   *
   * @return Expected edge cut of the partition.
   */
  dist::GlobalEdgeWeight compute_partition(dist::BlockID k, std::span<dist::BlockID> partition);

  /*!
   * Partitions the graph set by `copy_graph()` or `set_graph()` into `k` blocks with
   * a maximum imbalance of `epsilon`.
   *
   * @param k Number of blocks.
   * @param epsilon Balance constraint (e.g., 0.03 for max 3% imbalance).
   * @param[out] partition Span of length `n` to store the partitioning.
   *
   * @return Expected edge cut of the partition.
   */
  dist::GlobalEdgeWeight
  compute_partition(dist::BlockID k, double epsilon, std::span<dist::BlockID> partition);

  /*!
   * Partitions the graph set by `copy_graph()` or `set_graph()` such that the
   * weight of each block is upper bounded by `max_block_weights`. The number of blocks is given
   * implicitly by the size of the vector.
   *
   * @param max_block_weights Maximum weight for each block of the partition.
   * @param[out] partition Span of length `n` to store the partitioning.
   *
   * @return Expected edge cut of the partition.
   */
  dist::GlobalEdgeWeight compute_partition(
      std::vector<dist::BlockWeight> max_block_weights, std::span<dist::BlockID> partition
  );

  /*!
   * Partitions the graph set by `copy_graph()` or `set_graph()` such that the
   * weight of each block is upper bounded by `max_block_weight_factors` times the total node weigh
   * of the graph. The number of blocks is given implicitly by the size of the vector.
   *
   * @param max_block_weight_factors Maximum weight factor for each block of the partition.
   * @param[out] partition Span of length `n` to store the partitioning.
   *
   * @return Expected edge cut of the partition.
   */
  dist::GlobalEdgeWeight compute_partition(
      std::vector<double> max_block_weight_factors, std::span<dist::BlockID> partition
  );

  const dist::DistributedGraph *graph() const;

private:
  dist::GlobalEdgeWeight compute_partition(std::span<dist::BlockID> partition);

  MPI_Comm _comm;
  int _num_threads;

  int _max_timer_depth = std::numeric_limits<int>::max();
  OutputLevel _output_level = OutputLevel::APPLICATION;
  dist::Context _ctx;

  std::unique_ptr<dist::DistributedGraph> _graph_ptr;
  tbb::global_control _gc;

  bool _was_rearranged = false;
};

} // namespace kaminpar

#endif
