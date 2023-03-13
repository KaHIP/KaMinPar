/*******************************************************************************
 * @file:   kaminpar.h
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 * @brief:  Public symbols of the shared-memory partitioner
 ******************************************************************************/
#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include <tbb/global_control.h>

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
#else  // KAMINPAR_64BIT_WEIGHTS
using NodeWeight = std::int32_t;
using EdgeWeight = std::int32_t;
#endif // KAMINPAR_64BIT_WEIGHTS

using BlockID = std::uint32_t;
using BlockWeight = NodeWeight;
using Gain = EdgeWeight;
using Degree = EdgeID;
using Clustering = std::vector<NodeID>;

constexpr BlockID kInvalidBlockID = std::numeric_limits<BlockID>::max();
constexpr NodeID kInvalidNodeID = std::numeric_limits<NodeID>::max();
constexpr EdgeID kInvalidEdgeID = std::numeric_limits<EdgeID>::max();
constexpr NodeWeight kInvalidNodeWeight =
    std::numeric_limits<NodeWeight>::max();
constexpr EdgeWeight kInvalidEdgeWeight =
    std::numeric_limits<EdgeWeight>::max();
constexpr BlockWeight kInvalidBlockWeight =
    std::numeric_limits<BlockWeight>::max();
constexpr Degree kMaxDegree = std::numeric_limits<Degree>::max();

enum class OutputLevel : std::uint8_t {
  QUIET,
  PROGRESS,
  APPLICATION,
  EXPERIMENT,
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

struct PartitionContext;

struct BlockWeightsContext {
  void setup(const PartitionContext &ctx);
  void setup(const PartitionContext &ctx, const std::vector<BlockID> &final_ks);

  [[nodiscard]] BlockWeight max(BlockID b) const;
  [[nodiscard]] const std::vector<BlockWeight> &all_max() const;
  [[nodiscard]] BlockWeight perfectly_balanced(BlockID b) const;
  [[nodiscard]] const std::vector<BlockWeight> &all_perfectly_balanced() const;

private:
  std::vector<BlockWeight> _perfectly_balanced_block_weights;
  std::vector<BlockWeight> _max_block_weights;
};

struct PartitionContext {
  PartitioningMode mode;
  double epsilon;
  BlockID k;

  BlockWeightsContext block_weights{};
  void setup_block_weights();

  NodeID n = kInvalidNodeID;
  EdgeID m = kInvalidEdgeID;
  NodeWeight total_node_weight = kInvalidNodeWeight;
  EdgeWeight total_edge_weight = kInvalidEdgeWeight;
  NodeWeight max_node_weight = kInvalidNodeWeight;
};

struct LabelPropagationCoarseningContext {
  std::size_t num_iterations;
  Degree large_degree_threshold;
  NodeID max_num_neighbors;
  double two_hop_clustering_threshold;

  [[nodiscard]] bool use_two_hop_clustering(const NodeID old_n,
                                            const NodeID new_n) const {
    return (1.0 - 1.0 * new_n / old_n) <= two_hop_clustering_threshold;
  }
};

struct CoarseningContext {
  ClusteringAlgorithm algorithm;
  LabelPropagationCoarseningContext lp;
  NodeID contraction_limit;
  bool enforce_contraction_limit;
  double convergence_threshold;
  ClusterWeightLimit cluster_weight_limit;
  double cluster_weight_multiplier;

  [[nodiscard]] inline bool
  coarsening_should_converge(const NodeID old_n, const NodeID new_n) const {
    return (1.0 - 1.0 * new_n / old_n) <= convergence_threshold;
  }
};

struct LabelPropagationRefinementContext {
  std::size_t num_iterations;
  Degree large_degree_threshold;
  NodeID max_num_neighbors;
};

struct FMRefinementContext {
  FMStoppingRule stopping_rule;
  NodeID num_fruitless_moves;
  double alpha;
  std::size_t num_iterations;
  double improvement_abortion_threshold;
};

struct BalancerRefinementContext {
  BalancingAlgorithm algorithm;
  BalancingTimepoint timepoint;
};

struct RefinementContext {
  RefinementAlgorithm algorithm;
  LabelPropagationRefinementContext lp;
  FMRefinementContext fm;
  BalancerRefinementContext balancer;
};

struct InitialPartitioningContext {
  CoarseningContext coarsening;
  RefinementContext refinement;
  InitialPartitioningMode mode;
  double repetition_multiplier;
  std::size_t min_num_repetitions;
  std::size_t min_num_non_adaptive_repetitions;
  std::size_t max_num_repetitions;
  std::size_t num_seed_iterations;
  bool use_adaptive_bipartitioner_selection;
  std::size_t multiplier_exponent;
};

struct ParallelContext {
  bool use_interleaved_numa_allocation;
  std::size_t num_threads;
};

struct Context {
  std::string graph_filename;
  int seed;
  bool save_partition;
  std::string partition_directory;
  std::string partition_filename;
  bool degree_weights;
  bool quiet;
  bool parsable_output;

  bool unchecked_io;
  bool validate_io;

  PartitionContext partition;
  CoarseningContext coarsening;
  InitialPartitioningContext initial_partitioning;
  RefinementContext refinement;
  ParallelContext parallel;

  [[nodiscard]] std::string partition_file() const {
    return partition_directory + "/" + partition_filename;
  }
};
} // namespace kaminpar::shm

namespace kaminpar::shm {
Context create_context_by_preset_name(const std::string &name);
Context create_default_context();
std::unordered_set<std::string> get_preset_names();

class KaMinPar {
public:
  KaMinPar(int num_threads, Context ctx);

  void set_output_level(OutputLevel output_level);

  void set_max_timer_depth(int max_timer_depth);

  Context &context();

  void import_graph(EdgeID *nodes, NodeID *edges, NodeWeight *node_weights,
                    EdgeWeight *edge_weights);

  NodeID load_graph(const std::string &filename);

  EdgeWeight compute_partition(int seed, BlockID k, BlockID *partition);

private:
  int _num_threads;

  int _max_timer_depth = std::numeric_limits<int>::max();
  OutputLevel _output_level = OutputLevel::APPLICATION;
  Context _ctx;

  std::unique_ptr<struct Graph> _graph_ptr;
  tbb::global_control _gc;

  bool _was_rearranged = false;
};
} // namespace kaminpar::shm
