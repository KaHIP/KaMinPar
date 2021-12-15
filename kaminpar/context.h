#pragma once

#include "datastructure/graph.h"
#include "definitions.h"

#include <cmath>
#include <map>
#include <string_view>

#define DECLARE_ENUM_STRING_CONVERSION(type_name, prefix_name)                                                         \
  type_name prefix_name##_from_string(const std::string &searched);                                                    \
  std::ostream &operator<<(std::ostream &os, const type_name &value);                                                  \
  std::string prefix_name##_names(const std::string &sep = ", ")

namespace kaminpar {
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

enum class FMStoppingRule {
  SIMPLE,
  ADAPTIVE,
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

DECLARE_ENUM_STRING_CONVERSION(ClusteringAlgorithm, clustering_algorithm);
DECLARE_ENUM_STRING_CONVERSION(RefinementAlgorithm, refinement_algorithm);
DECLARE_ENUM_STRING_CONVERSION(BalancingTimepoint, balancing_timepoint);
DECLARE_ENUM_STRING_CONVERSION(BalancingAlgorithm, balancing_algorithm);
DECLARE_ENUM_STRING_CONVERSION(FMStoppingRule, fm_stopping_rule);
DECLARE_ENUM_STRING_CONVERSION(PartitioningMode, partitioning_mode);
DECLARE_ENUM_STRING_CONVERSION(ClusterWeightLimit, cluster_weight_limit);
DECLARE_ENUM_STRING_CONVERSION(InitialPartitioningMode, initial_partitioning_mode);

struct PartitionContext {
  PartitioningMode mode{PartitioningMode::DEEP};
  double epsilon{0.03};
  BlockID k{0};
  bool remove_isolated_nodes{true};
  bool fast_initial_partitioning{false};

  void setup(const Graph &graph);
  void setup_max_block_weight();
  void setup_max_block_weight(const scalable_vector<BlockID> &final_ks);

  [[nodiscard]] NodeWeight max_block_weight(BlockID b) const;
  [[nodiscard]] const scalable_vector<NodeWeight> &max_block_weights() const;
  [[nodiscard]] NodeWeight perfectly_balanced_block_weight(BlockID b) const;
  [[nodiscard]] const scalable_vector<NodeWeight> &perfectly_balanced_block_weights() const;

  void print(std::ostream &out, const std::string &prefix = "") const;

  NodeID n{};
  EdgeID m{};
  NodeWeight total_node_weight{};
  EdgeWeight total_edge_weight{};
  NodeWeight max_node_weight{};

  void reset_block_weights();

  scalable_vector<NodeWeight> _perfectly_balanced_block_weights{};
  scalable_vector<NodeWeight> _max_block_weights{};
};

struct LabelPropagationCoarseningContext {
  //! Perform at most this many iterations.
  std::size_t num_iterations{0};

  //! Ignore nodes whose degree exceeds this threshold.
  Degree large_degree_threshold{kMaxDegree};

  //! If the number of clusters shrunk by less than this factor, try to merge singleton clusters that favor the same
  //! cluster.
  double merge_nonadjacent_clusters_threshold{0.5};

  //! In case of `merge_nonadjacent_clusters_threshold`, also merge isolated singleton clusters.
  bool merge_singleton_clusters{true};

  //! Maximum number of neighbors to scan before assigning a node to a cluster. 0 = always scan all neighbors.
  NodeID max_num_neighbors{std::numeric_limits<NodeID>::max()};

  void print(std::ostream &out, const std::string &prefix = "") const;

  [[nodiscard]] bool should_merge_nonadjacent_clusters(const NodeID old_n, const NodeID new_n) const {
    return (1.0 - 1.0 * new_n / old_n) <= merge_nonadjacent_clusters_threshold;
  }
};

struct CoarseningContext {
  //! Clustering algorithm, e.g., label propagation.
  ClusteringAlgorithm algorithm{ClusteringAlgorithm::NOOP};
  LabelPropagationCoarseningContext lp{};

  //! Abort coarsening once the number of nodes falls below `2 * contraction_limit`.
  NodeID contraction_limit{0};

  //! Control the clustering algorithm to enforce the contraction limit.
  bool enforce_contraction_limit{false};

  //! We terminate coarsening once the graph shrunk by at most this factor.
  double convergence_threshold{0.0};

  //! The rule that use to compute the maximum cluster weight.
  ClusterWeightLimit cluster_weight_limit{ClusterWeightLimit::EPSILON_BLOCK_WEIGHT};

  //! Multiplicative factor to the maximum cluster weight limit computed.
  double cluster_weight_multiplier{1.0};

  void print(std::ostream &out, const std::string &prefix = "") const;

  [[nodiscard]] inline bool should_converge(const NodeID old_n, const NodeID new_n) const {
    return (1.0 - 1.0 * new_n / old_n) <= convergence_threshold;
  }
};

struct LabelPropagationRefinementContext {
  //! Perform at most this many iterations.
  std::size_t num_iterations{0};

  //! Ignore nodes whose degree exceeds this threshold.
  Degree large_degree_threshold{kMaxDegree};

  //! Maximum number of neighbors to scan before assigning a node to a block. 0 = always scan all neighbors.
  NodeID max_num_neighbors{std::numeric_limits<NodeID>::max()};

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct FMRefinementContext {
  FMStoppingRule stopping_rule{FMStoppingRule::SIMPLE};
  NodeID num_fruitless_moves{100};
  double alpha{1.0};
  std::size_t num_iterations{5};
  double improvement_abortion_threshold{0.0001};

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct BalancerRefinementContext {
  BalancingAlgorithm algorithm{BalancingAlgorithm::BLOCK_LEVEL_PARALLEL_BALANCER};
  BalancingTimepoint timepoint{BalancingTimepoint::BEFORE_KWAY_REFINEMENT};

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct RefinementContext {
  RefinementAlgorithm algorithm{RefinementAlgorithm::NOOP};
  LabelPropagationRefinementContext lp;
  FMRefinementContext fm;
  BalancerRefinementContext balancer;

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct InitialPartitioningContext {
  CoarseningContext coarsening;
  RefinementContext refinement;
  InitialPartitioningMode mode;
  double repetition_multiplier{1.0};
  std::size_t min_num_repetitions{10};
  std::size_t min_num_non_adaptive_repetitions{5};
  std::size_t max_num_repetitions{50};
  std::size_t num_seed_iterations{1};
  bool use_adaptive_epsilon{true};
  bool use_adaptive_bipartitioner_selection{true};
  std::size_t multiplier_exponent{0};
  bool parallelize_bisections{false};

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct DebugContext {
  bool just_sanitize_args{false};
  bool force_clean_build{false};

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct ParallelContext {
  bool use_interleaved_numa_allocation{true};
  std::size_t num_threads{1};

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct Context {
  std::string graph_filename{};
  int seed{0};
  bool save_partition{false};
  std::string partition_directory{"./"};
  std::string partition_filename{};
  bool ignore_weights{false};
  bool show_local_timers{false};
  bool quiet{false};

  PartitionContext partition{};
  CoarseningContext coarsening{};
  InitialPartitioningContext initial_partitioning{};
  RefinementContext refinement{};
  DebugContext debug{};
  ParallelContext parallel{};

  void print(std::ostream &out, const std::string &prefix = "") const;

  void setup(const Graph &graph);
  [[nodiscard]] std::string partition_file() const { return partition_directory + "/" + partition_filename; }
};

std::ostream &operator<<(std::ostream &out, const Context &context);

Context create_default_context();
Context create_default_context(const Graph &graph, BlockID k, double epsilon);

PartitionContext create_bipartition_context(const PartitionContext &k_p_ctx, const Graph &subgraph, BlockID final_k1,
                                            BlockID final_k2);

double compute_2way_adaptive_epsilon(const PartitionContext &p_ctx, NodeWeight subgraph_total_node_weight,
                                     BlockID subgraph_final_k);

NodeWeight compute_max_cluster_weight(const Graph &c_graph, const PartitionContext &input_p_ctx,
                                      const CoarseningContext &c_ctx);

} // namespace kaminpar
