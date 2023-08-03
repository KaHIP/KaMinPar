/*******************************************************************************
 * Public interface of the distributed partitioner.
 *
 * @file:   dkaminpar.h
 * @author: Daniel Seemaier
 * @date:   30.01.2023
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>

#include <mpi.h>
#include <tbb/global_control.h>

#include "kaminpar/kaminpar.h"

namespace kaminpar::mpi {
using PEID = int;
}

namespace kaminpar::dist {
using GlobalNodeID = std::uint64_t;
using GlobalNodeWeight = std::int64_t;
using GlobalEdgeID = std::uint64_t;
using GlobalEdgeWeight = std::int64_t;
using BlockWeight = std::int64_t;

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

enum class GlobalClusteringAlgorithm {
  NOOP,
  LP,
  HEM,
  HEM_LP,
};

enum class LocalClusteringAlgorithm {
  NOOP,
  LP,
};

enum class InitialPartitioningAlgorithm {
  KAMINPAR,
  MTKAHYPAR,
  RANDOM,
};

enum class RefinementAlgorithm {
  NOOP,
  BATCHED_LP,
  COLORED_LP,
  LOCAL_FM,
  GLOBAL_FM,
  JET_REFINER,
  JET_BALANCER,
  GREEDY_NODE_BALANCER,
  GREEDY_CLUSTER_BALANCER,
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

struct ParallelContext {
  std::size_t num_threads;
  std::size_t num_mpis;
};

struct LabelPropagationCoarseningContext {
  int num_iterations;
  NodeID passive_high_degree_threshold;
  NodeID active_high_degree_threshold;
  NodeID max_num_neighbors;
  bool merge_singleton_clusters;
  double merge_nonadjacent_clusters_threshold;
  int total_num_chunks;
  int fixed_num_chunks;
  int min_num_chunks;
  bool ignore_ghost_nodes;
  bool keep_ghost_clusters;
  bool scale_chunks_with_threads;

  bool sync_cluster_weights;
  bool enforce_cluster_weights;
  bool cheap_toplevel;

  bool prevent_cyclic_moves;
  bool enforce_legacy_weight;

  bool should_merge_nonadjacent_clusters(NodeID old_n, NodeID new_n) const;
  int compute_num_chunks(const ParallelContext &parallel) const;
};

struct HEMCoarseningContext {
  int max_num_coloring_chunks;
  int fixed_num_coloring_chunks;
  int min_num_coloring_chunks;
  bool scale_coloring_chunks_with_threads;
  double small_color_blacklist;
  bool only_blacklist_input_level;
  bool ignore_weight_limit;

  int compute_num_coloring_chunks(const ParallelContext &parallel) const;
};

struct ColoredLabelPropagationRefinementContext {
  int num_iterations;
  int num_move_execution_iterations;
  int num_probabilistic_move_attempts;
  bool sort_by_rel_gain;

  int max_num_coloring_chunks;
  int fixed_num_coloring_chunks;
  int min_num_coloring_chunks;
  bool scale_coloring_chunks_with_threads;
  double small_color_blacklist;
  bool only_blacklist_input_level;

  bool track_local_block_weights;
  bool use_active_set;

  LabelPropagationMoveExecutionStrategy move_execution_strategy;

  int compute_num_coloring_chunks(const ParallelContext &parallel) const;
};

struct LabelPropagationRefinementContext {
  NodeID active_high_degree_threshold;
  int num_iterations;

  int total_num_chunks;
  int fixed_num_chunks;
  int min_num_chunks;

  int num_move_attempts;
  bool ignore_probabilities;
  bool scale_chunks_with_threads;

  int compute_num_chunks(const ParallelContext &parallel) const;
};

struct FMRefinementContext {
  double alpha;
  NodeID radius;
  PEID pe_radius;
  bool overlap_regions;
  std::size_t num_iterations;
  bool sequential;
  bool premove_locally;
  NodeID bound_degree;
  bool contract_border;

  int max_hops;
  int max_radius;

  int num_global_iterations;
  int num_local_iterations;

  bool revert_local_moves_after_batch;
};

struct CoarseningContext {
  // Global clustering
  std::size_t max_global_clustering_levels;
  GlobalClusteringAlgorithm global_clustering_algorithm;
  LabelPropagationCoarseningContext global_lp;
  HEMCoarseningContext hem;

  // Local clustering
  std::size_t max_local_clustering_levels;
  LocalClusteringAlgorithm local_clustering_algorithm;
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

struct GreedyBalancerContext {
  int max_num_rounds;
  bool enable_strong_balancing;
  NodeID num_nodes_per_block;
  bool enable_fast_balancing;
  double fast_balancing_threshold;
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

  ClusterSizeStrategy cluster_size_strategy;
  double cluster_size_multiplier;

  ClusterStrategy cluster_strategy;
  int cluster_rebuild_interval;
};

struct JetBalancerContext {
  int num_weak_iterations;
  int num_strong_iterations;
};

struct JetRefinementContext {
  int num_iterations;
  double min_c;
  double max_c;
  bool interpolate_c;
  bool use_abortion_threshold;
  double abortion_threshold;
  RefinementAlgorithm balancing_algorithm;
};

struct RefinementContext {
  std::vector<RefinementAlgorithm> algorithms;
  bool refine_coarsest_level;

  LabelPropagationRefinementContext lp;
  ColoredLabelPropagationRefinementContext colored_lp;
  FMRefinementContext fm;
  GreedyBalancerContext greedy_balancer;
  ClusterBalancerContext cluster_balancer;

  JetRefinementContext jet;
  JetBalancerContext jet_balancer;

  bool includes_algorithm(RefinementAlgorithm algorithm) const;
};

struct PartitionContext {
  PartitionContext(BlockID k, BlockID K, double epsilon);

  PartitionContext(const PartitionContext &other);
  PartitionContext &operator=(const PartitionContext &other);

  ~PartitionContext();

  BlockID k = kInvalidBlockID;
  BlockID K = kInvalidBlockID;
  double epsilon;

  std::unique_ptr<struct GraphContext> graph;
};

struct DebugContext {
  std::string graph_filename;
  bool save_coarsest_graph;
  bool save_coarsest_partition;
};

struct Context {
  GraphOrdering rearrange_by;

  PartitioningMode mode;

  bool enable_pe_splitting;
  bool simulate_singlethread;

  PartitionContext partition;
  ParallelContext parallel;
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
  ~dKaMinPar();

  void set_output_level(OutputLevel output_level);

  void set_max_timer_depth(int max_timer_depth);

  dist::Context &context();

  void import_graph(
      dist::GlobalNodeID *node_distribution,
      dist::GlobalEdgeID *nodes,
      dist::GlobalNodeID *edges,
      dist::GlobalNodeWeight *node_weights,
      dist::GlobalEdgeWeight *edge_weights
  );

  dist::GlobalEdgeWeight compute_partition(int seed, dist::BlockID k, dist::BlockID *partition);

private:
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
