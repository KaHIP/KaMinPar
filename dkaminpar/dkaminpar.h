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

enum class KWayRefinementAlgorithm {
  NOOP,
  LP,
  LOCAL_FM,
  FM,
  COLORED_LP,
  GREEDY_BALANCER,
  JET,
  MOVE_SET_BALANCER,
  JET_BALANCER,
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
  std::size_t num_mpis = 0;
};

struct LabelPropagationCoarseningContext {
  int num_iterations = 0;
  NodeID passive_high_degree_threshold = 0;
  NodeID active_high_degree_threshold = 0;
  NodeID max_num_neighbors = 0;
  bool merge_singleton_clusters = 0;
  double merge_nonadjacent_clusters_threshold = 0;
  int total_num_chunks = 0;
  int fixed_num_chunks = 0;
  int min_num_chunks = 0;
  bool ignore_ghost_nodes = false;
  bool keep_ghost_clusters = false;
  bool scale_chunks_with_threads = false;

  bool sync_cluster_weights = false;
  bool enforce_cluster_weights = false;
  bool cheap_toplevel = false;

  bool prevent_cyclic_moves = false;
  bool enforce_legacy_weight = false;

  bool should_merge_nonadjacent_clusters(NodeID old_n, NodeID new_n) const;
  int compute_num_chunks(const ParallelContext &parallel) const;
};

struct HEMCoarseningContext {
  int max_num_coloring_chunks = 0;
  int fixed_num_coloring_chunks = 0;
  int min_num_coloring_chunks = 0;
  bool scale_coloring_chunks_with_threads = false;
  double small_color_blacklist = 0;
  bool only_blacklist_input_level = false;
  bool ignore_weight_limit = false;

  int compute_num_coloring_chunks(const ParallelContext &parallel) const;
};

struct ColoredLabelPropagationRefinementContext {
  int num_iterations = 0;
  int num_move_execution_iterations = 0;
  int num_probabilistic_move_attempts = 0;
  bool sort_by_rel_gain = false;

  int max_num_coloring_chunks = 0;
  int fixed_num_coloring_chunks = 0;
  int min_num_coloring_chunks = 0;
  bool scale_coloring_chunks_with_threads = false;
  double small_color_blacklist = 0;
  bool only_blacklist_input_level = false;

  bool track_local_block_weights = false;
  bool use_active_set = false;

  LabelPropagationMoveExecutionStrategy move_execution_strategy =
      LabelPropagationMoveExecutionStrategy::PROBABILISTIC;

  int compute_num_coloring_chunks(const ParallelContext &parallel) const;
};

struct LabelPropagationRefinementContext {
  NodeID active_high_degree_threshold = 0;
  int num_iterations = 0;

  int total_num_chunks = 0;
  int fixed_num_chunks = 0;
  int min_num_chunks = 0;

  int num_move_attempts = 0;
  bool ignore_probabilities = false;
  bool scale_chunks_with_threads = false;

  int compute_num_chunks(const ParallelContext &parallel) const;
};

struct FMRefinementContext {
  double alpha = 0.0;
  NodeID radius = 0;
  PEID pe_radius = 0;
  bool overlap_regions = false;
  std::size_t num_iterations = 0;
  bool sequential = false;
  bool premove_locally = false;
  NodeID bound_degree = 0;
  bool contract_border = false;
};

struct CoarseningContext {
  // Global clustering
  std::size_t max_global_clustering_levels = 0;
  GlobalClusteringAlgorithm global_clustering_algorithm;
  LabelPropagationCoarseningContext global_lp;
  HEMCoarseningContext hem;

  // Local clustering
  std::size_t max_local_clustering_levels = 0;
  LocalClusteringAlgorithm local_clustering_algorithm;
  LabelPropagationCoarseningContext local_lp;

  // Cluster weight limit
  NodeID contraction_limit = 0;
  shm::ClusterWeightLimit cluster_weight_limit = shm::ClusterWeightLimit::EPSILON_BLOCK_WEIGHT;
  double cluster_weight_multiplier = 0.0;

  // Graph contraction
  double max_cnode_imbalance = std::numeric_limits<double>::max();
  bool migrate_cnode_prefix = true;
  bool force_perfect_cnode_balance = false;

  void setup(const ParallelContext &parallel);
};

struct InitialPartitioningContext {
  InitialPartitioningAlgorithm algorithm;
  shm::Context kaminpar;
};

struct GreedyBalancerContext {
  int max_num_rounds = 0;
  bool enable_strong_balancing = false;
  NodeID num_nodes_per_block = 0;
  bool enable_fast_balancing = false;
  double fast_balancing_threshold = 0.0;
};

struct JetBalancerContext {
  int num_weak_iterations = 0;
  int num_strong_iterations = 0;
};

struct JetRefinementContext {
  int num_iterations = 0;
  double min_c = 0.0;
  double max_c = 0.0;
  bool interpolate_c = false;
  bool use_abortion_threshold = false;
  double abortion_threshold = 0;
};

struct RefinementContext {
  std::vector<KWayRefinementAlgorithm> algorithms;
  bool refine_coarsest_level = false;

  LabelPropagationRefinementContext lp;
  ColoredLabelPropagationRefinementContext colored_lp;
  FMRefinementContext fm;
  GreedyBalancerContext greedy_balancer;

  JetRefinementContext jet;
  JetBalancerContext jet_balancer;

  bool includes_algorithm(KWayRefinementAlgorithm algorithm) const;
};

struct PartitionContext {
  PartitionContext(BlockID k, BlockID K, double epsilon);

  PartitionContext(const PartitionContext &other);
  PartitionContext &operator=(const PartitionContext &other);

  ~PartitionContext();

  BlockID k = kInvalidBlockID;
  BlockID K = kInvalidBlockID;
  double epsilon = 0.0;

  std::unique_ptr<struct GraphContext> graph;
};

struct DebugContext {
  std::string graph_filename = "";
  bool save_coarsest_graph = false;
  bool save_coarsest_partition = false;
};

struct Context {
  GraphOrdering rearrange_by = GraphOrdering::NATURAL;

  PartitioningMode mode = PartitioningMode::DEEP;

  bool enable_pe_splitting = false;
  bool simulate_singlethread = false;

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
