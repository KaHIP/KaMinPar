/*******************************************************************************
 * @file:   kaminpar.h
 * @author: Daniel Seemaier
 * @date:   13.03.2023
 * @brief:  Public shared-memory partitioning interface.
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

namespace kaminpar {
enum class OutputLevel : std::uint8_t {
  QUIET,       //! Disable all output to stdout.
  PROGRESS,    //! Continuously output progress information while partitioning.
  APPLICATION, //! Also output the application banner and context summary.
  EXPERIMENT,  //! Also output information only relevant for benchmarking.
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
#else  // KAMINPAR_64BIT_WEIGHTS
using NodeWeight = std::int32_t;
using EdgeWeight = std::int32_t;
#endif // KAMINPAR_64BIT_WEIGHTS

using BlockID = std::uint32_t;
using BlockWeight = NodeWeight;

constexpr BlockID kInvalidBlockID = std::numeric_limits<BlockID>::max();
constexpr NodeID kInvalidNodeID = std::numeric_limits<NodeID>::max();
constexpr EdgeID kInvalidEdgeID = std::numeric_limits<EdgeID>::max();
constexpr NodeWeight kInvalidNodeWeight = std::numeric_limits<NodeWeight>::max();
constexpr EdgeWeight kInvalidEdgeWeight = std::numeric_limits<EdgeWeight>::max();
constexpr BlockWeight kInvalidBlockWeight = std::numeric_limits<BlockWeight>::max();

//
// Coarsening
//

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

struct LabelPropagationCoarseningContext {
  std::size_t num_iterations;
  NodeID large_degree_threshold;
  NodeID max_num_neighbors;
  double two_hop_clustering_threshold;

  [[nodiscard]] bool use_two_hop_clustering(const NodeID old_n, const NodeID new_n) const {
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

struct LabelPropagationRefinementContext {
  std::size_t num_iterations;
  NodeID large_degree_threshold;
  NodeID max_num_neighbors;
};

struct KwayFMRefinementContext {
  NodeID num_seed_nodes;
  double alpha;
  int num_iterations;
  bool unlock_seed_nodes;
  bool use_exact_abortion_threshold;
  double abortion_threshold;

  bool dbg_compute_batch_size_statistics;
};

struct JetRefinementContext {
  int num_iterations;
  double interpolate_c;
  double min_c;
  double max_c;
  double abortion_threshold;
};

struct MtKaHyParRefinementContext {
  std::string config_filename;
};

struct BalancerRefinementContext {};

struct RefinementContext {
  std::vector<RefinementAlgorithm> algorithms;
  LabelPropagationRefinementContext lp;
  KwayFMRefinementContext kway_fm;
  BalancerRefinementContext balancer;
  JetRefinementContext jet;
  MtKaHyParRefinementContext mtkahypar;

  bool includes_algorithm(const RefinementAlgorithm algorithm) const {
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

struct InitialPartitioningContext {
  InitialPartitioningMode mode;

  InitialCoarseningContext coarsening;
  InitialRefinementContext refinement;

  double repetition_multiplier;
  std::size_t min_num_repetitions;
  std::size_t min_num_non_adaptive_repetitions;
  std::size_t max_num_repetitions;
  std::size_t num_seed_iterations;
  bool use_adaptive_bipartitioner_selection;
  std::size_t multiplier_exponent;
};

//
// Application level
//

class Graph;
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
  double epsilon;
  BlockID k;

  BlockWeightsContext block_weights{};
  void setup_block_weights();

  NodeID n = kInvalidNodeID;
  EdgeID m = kInvalidEdgeID;
  NodeWeight total_node_weight = kInvalidNodeWeight;
  EdgeWeight total_edge_weight = kInvalidEdgeWeight;
  NodeWeight max_node_weight = kInvalidNodeWeight;

  void setup(const Graph &graph);
};

struct ParallelContext {
  int num_threads;
};

struct DebugContext {
  std::string graph_name;
  bool dump_coarsest_graph;
  bool dump_coarsest_partition;
  bool dump_graph_hierarchy;
  bool dump_partition_hierarchy;
};

enum class PartitioningMode {
  DEEP,
  RB,
};

struct Context {
  PartitioningMode mode;

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
Context create_fast_context();
Context create_default_context();
Context create_largek_context();
Context create_strong_context();
Context create_jet_context();
} // namespace kaminpar::shm

//
// Shared-memory partitioner interface
//

namespace kaminpar {
class KaMinPar {
public:
  KaMinPar(int num_threads, shm::Context ctx);
  ~KaMinPar();

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
   * Sets the graph to be partitioned by taking ownership of the given pointers. In particular, the
   * partitioner might modify the data pointed to.
   * The caller is responsible for free'ing the memory.
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
  void take_graph(
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
      shm::EdgeID *const xadj,
      shm::NodeID *const adjncy,
      shm::NodeWeight *const vwgt,
      shm::EdgeWeight *const adjwgt
  );

  /*!
   * Partitions the graph set by `take_graph()` or `copy_graph()` into `k` blocks.
   *
   * @param seed The seed for the random number generator. Note that when using more than one
   * thread, partitioning is non-deterministic even with a fixed seed.
   * @param k The number of blocks to partition the graph into.
   * @param partition Array of length `n` for storing the partition. The caller is reponsible for
   * allocating and freeing the memory.
   *
   * @return The edge-cut of the partition.
   */
  shm::EdgeWeight compute_partition(int seed, shm::BlockID k, shm::BlockID *partition);

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
