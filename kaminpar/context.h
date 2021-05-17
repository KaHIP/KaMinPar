#pragma once

#include "datastructure/graph.h"
#include "definitions.h"

#include <cmath>
#include <map>
#include <ranges>
#include <string_view>

#define DECLARE_ENUM_STRING_CONVERSION(type_name, prefix_name)                                                         \
  type_name prefix_name##_from_string(const std::string &searched);                                                    \
  std::ostream &operator<<(std::ostream &os, const type_name &value);                                                  \
  std::string prefix_name##_names(const std::string &sep = ", ")

namespace kaminpar {
enum class CoarseningAlgorithm {
  NOOP,
  PARALLEL_LABEL_PROPAGATION,
};

enum class RefinementAlgorithm {
  PARALLEL_LABEL_PROPAGATION,
  TWO_WAY_FM,
  NOOP,
};

enum class BalancingTimepoint {
  BEFORE_KWAY_REFINEMENT,
  AFTER_KWAY_REFINEMENT,
  BEFORE_AND_AFTER_KWAY_REFINEMENT,
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

DECLARE_ENUM_STRING_CONVERSION(CoarseningAlgorithm, coarsening_algorithm);
DECLARE_ENUM_STRING_CONVERSION(RefinementAlgorithm, refinement_algorithm);
DECLARE_ENUM_STRING_CONVERSION(BalancingTimepoint, balancing_timepoint);
DECLARE_ENUM_STRING_CONVERSION(BalancingAlgorithm, balancing_algorithm);
DECLARE_ENUM_STRING_CONVERSION(FMStoppingRule, fm_stopping_rule);
DECLARE_ENUM_STRING_CONVERSION(PartitioningMode, partitioning_mode);

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

  NodeID n;
  EdgeID m;
  NodeWeight total_node_weight;
  EdgeWeight total_edge_weight;
  NodeWeight max_node_weight;

private:
  void reset_block_weights();

  scalable_vector<NodeWeight> _perfectly_balanced_block_weights{};
  scalable_vector<NodeWeight> _max_block_weights{};
};

struct CoarseningContext {
  //
  // General options
  //

  bool enable{true};

  // Coarsening algorithm, e.g. label propagation
  CoarseningAlgorithm algorithm{CoarseningAlgorithm::NOOP};
  // Constant C, coarsen until 2C nodes
  NodeID contraction_limit{0};
  // If set, use C' = min{C, n/k} (ensure linear asymptotic runtime)
  bool use_adaptive_contraction_limit{false};
  // Abort coarsening if the graph shrunk by less than this factor after the last iteration
  double shrink_factor_abortion_threshold{0};

  //
  // Max cluster weight options
  //

  // If set, base adaptive + block weight based max cluster weight on adapted epsilon
  bool use_adaptive_epsilon{false};
  // Multiplier for adaptive cluster weight (set to 0 to disable)
  double adaptive_cluster_weight_multiplier{1.0};
  // If >0, ensure that a graph can shrink by at most this factor between two levels of the graph hierarchy
  double min_shrink_factor{0.0};
  // If set, set max cluster weight to max block weight / this factor
  double block_based_cluster_weight_factor{std::numeric_limits<double>::max()};

  //
  // Label propagation options
  //

  // Max number of iterations to perform
  std::size_t num_iterations{0};

  // Ignore nodes with degree larger than this (only an approximation, the algorithm skips buckets if the minimum
  // degree in the bucket is larger than this)
  Degree large_degree_threshold{1000000};

  double nonadjacent_clustering_fraction_threshold{2.0};

  bool randomize_chunk_order{true};

  bool merge_singleton_clusters{false};

  void print(std::ostream &out, const std::string &prefix = "") const;
};

struct LabelPropagationRefinementContext {
  std::size_t num_iterations{0};
  Degree large_degree_threshold{kMaxDegree};
  bool randomize_chunk_order{true};

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
  double repetition_multiplier{1.0};
  std::size_t min_num_repetitions{10};
  std::size_t min_num_non_adaptive_repetitions{5};
  std::size_t max_num_repetitions{50};
  std::size_t num_seed_iterations{1};
  bool use_adaptive_epsilon{true};
  bool use_adaptive_bipartitioner_selection{true};
  bool parallelize{true};
  bool parallelize_synchronized{true};
  std::size_t multiplier_exponent{0};

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
  static Context create_default();
  static Context create_default_for(const Graph &graph, BlockID k, double epsilon = 0.03);

  std::string graph_filename{""};
  int seed{0};
  bool save_partition{false};
  std::string partition_directory{"./"};
  std::string partition_filename{""};
  bool ignore_weights{false};
  bool show_local_timers{false};

  PartitionContext partition{};
  CoarseningContext coarsening{};
  InitialPartitioningContext initial_partitioning{};
  RefinementContext refinement{};
  DebugContext debug{};
  ParallelContext parallel{};

  void print(std::ostream &out, const std::string &prefix = "") const;

  void setup(const Graph &graph);
  [[nodiscard]] std::string partition_file() const { return partition_directory + "/" + partition_filename; }

  PartitionContext create_bipartition_partition_context(const Graph &subgraph, const BlockID final_k1,
                                                        const BlockID final_k2) const;
};

std::ostream &operator<<(std::ostream &out, const Context &context);

double compute_2way_adaptive_epsilon(const PartitionContext &p_ctx, const NodeWeight subgraph_total_node_weight, const BlockID subgraph_final_k);

NodeWeight compute_max_cluster_weight(const Graph &c_graph, const PartitionContext &input_p_ctx,
                                      const CoarseningContext &c_ctx, const CoarseningContext &input_c_ctx);

} // namespace kaminpar
