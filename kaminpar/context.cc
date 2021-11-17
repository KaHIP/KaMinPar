/*******************************************************************************
 * @file:   context.cc
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Configuration struct for KaMinPar.
 ******************************************************************************/
#include "kaminpar/context.h"

#include "kaminpar/utility/math.h"

namespace kaminpar {
using namespace std::string_literals;

//
// Define std::string <-> enum conversion functions
//

DEFINE_ENUM_STRING_CONVERSION(ClusteringAlgorithm, clustering_algorithm) = {
    {ClusteringAlgorithm::NOOP, "noop"},
    {ClusteringAlgorithm::LABEL_PROPAGATION, "lp"},
};

DEFINE_ENUM_STRING_CONVERSION(RefinementAlgorithm, refinement_algorithm) = {
    {RefinementAlgorithm::TWO_WAY_FM, "2way-fm"}, //
    {RefinementAlgorithm::LABEL_PROPAGATION, "lp"},
    {RefinementAlgorithm::NOOP, "noop"}, //
};

DEFINE_ENUM_STRING_CONVERSION(BalancingTimepoint, balancing_timepoint) = {
    {BalancingTimepoint::BEFORE_KWAY_REFINEMENT, "before-kway-ref"},
    {BalancingTimepoint::AFTER_KWAY_REFINEMENT, "after-kway-ref"},
    {BalancingTimepoint::ALWAYS, "always"},
    {BalancingTimepoint::NEVER, "never"},
};

DEFINE_ENUM_STRING_CONVERSION(BalancingAlgorithm, balancing_algorithm) = {
    {BalancingAlgorithm::BLOCK_LEVEL_PARALLEL_BALANCER, "block-parallel-balancer"},
};

DEFINE_ENUM_STRING_CONVERSION(FMStoppingRule, fm_stopping_rule) = {
    {FMStoppingRule::SIMPLE, "simple"},
    {FMStoppingRule::ADAPTIVE, "adaptive"},
};

DEFINE_ENUM_STRING_CONVERSION(PartitioningMode, partitioning_mode) = {
    {PartitioningMode::DEEP, "deep"},
    {PartitioningMode::RB, "rb"},
};

DEFINE_ENUM_STRING_CONVERSION(ClusterWeightLimit, cluster_weight_limit) = {
    {ClusterWeightLimit::EPSILON_BLOCK_WEIGHT, "epsilon-block-weight"},
    {ClusterWeightLimit::BLOCK_WEIGHT, "static-block-weight"},
    {ClusterWeightLimit::ONE, "one"},
    {ClusterWeightLimit::ZERO, "zero"},
};

DEFINE_ENUM_STRING_CONVERSION(InitialPartitioningMode, initial_partitioning_mode) = {
    {InitialPartitioningMode::SEQUENTIAL, "sequential"},
    {InitialPartitioningMode::ASYNCHRONOUS_PARALLEL, "async-parallel"},
    {InitialPartitioningMode::SYNCHRONOUS_PARALLEL, "sync-parallel"},
};

//
// PartitionContext
//

void PartitionContext::setup(const Graph &graph) {
  n = graph.n();
  m = graph.m();
  total_node_weight = graph.total_node_weight();
  total_edge_weight = graph.total_edge_weight();
  max_node_weight = graph.max_node_weight();
}

void PartitionContext::setup_max_block_weight() {
  ASSERT(k != kInvalidBlockID);

  const NodeWeight perfectly_balanced_block_weight = std::ceil(1.0 * total_node_weight / k);
  const NodeWeight max_block_weight = (1.0 + epsilon) * perfectly_balanced_block_weight;

  reset_block_weights();
  for (BlockID b = 0; b < k; ++b) {
    _perfectly_balanced_block_weights.push_back(perfectly_balanced_block_weight);
    if (max_node_weight == 1) { // don't relax balance constraint on output level
      _max_block_weights.push_back(max_block_weight);
    } else {
      _max_block_weights.push_back(
          std::max<NodeWeight>(max_block_weight, perfectly_balanced_block_weight + max_node_weight));
    }
  }
}

void PartitionContext::setup_max_block_weight(const scalable_vector<BlockID> &final_ks) {
  ASSERT(k == final_ks.size());

  const BlockID final_k = std::accumulate(final_ks.begin(), final_ks.end(), 0);
  const double block_weight = 1.0 * total_node_weight / final_k;

  reset_block_weights();
  for (BlockID b = 0; b < final_ks.size(); ++b) {
    _perfectly_balanced_block_weights.push_back(std::ceil(final_ks[b] * block_weight));
    if (max_node_weight == 1) { // don't relax balance constraint on output level
      _max_block_weights.push_back((1.0 + epsilon) * _perfectly_balanced_block_weights[b]);
    } else {
      _max_block_weights.push_back(std::max<NodeWeight>((1.0 + epsilon) * _perfectly_balanced_block_weights[b],
                                                        _perfectly_balanced_block_weights[b] + max_node_weight));
    }
  }
}

[[nodiscard]] NodeWeight PartitionContext::max_block_weight(const BlockID b) const {
  ASSERT(b < _max_block_weights.size());
  return _max_block_weights[b];
}

[[nodiscard]] const scalable_vector<NodeWeight> &PartitionContext::max_block_weights() const {
  return _max_block_weights;
}

[[nodiscard]] NodeWeight PartitionContext::perfectly_balanced_block_weight(const BlockID b) const {
  ASSERT(b < _perfectly_balanced_block_weights.size());
  return _perfectly_balanced_block_weights[b];
}

[[nodiscard]] const scalable_vector<NodeWeight> &PartitionContext::perfectly_balanced_block_weights() const {
  return _perfectly_balanced_block_weights;
}

void PartitionContext::reset_block_weights() {
  _perfectly_balanced_block_weights.clear();
  _perfectly_balanced_block_weights.reserve(k);
  _max_block_weights.clear();
  _max_block_weights.reserve(k);
}

//
// print() member functions
//

void PartitionContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "mode=" << mode << " "                                            //
      << prefix << "epsilon=" << epsilon << " "                                      //
      << prefix << "k=" << k << " "                                                  //
      << prefix << "fast_initial_partitioning=" << fast_initial_partitioning << " "; //
}

void CoarseningContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "graphutils=" << algorithm << " "                                  //
      << prefix << "contraction_limit=" << contraction_limit << " "                  //
      << prefix << "enforce_contraction_limit=" << enforce_contraction_limit << " "  //
      << prefix << "convergence_threshold=" << convergence_threshold << " "          //
      << prefix << "cluster_weight_limit=" << cluster_weight_limit << " "            //
      << prefix << "cluster_weight_multiplier=" << cluster_weight_multiplier << " "; //
  lp.print(out, prefix + "lp.");
}

void LabelPropagationCoarseningContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "num_iterations=" << num_iterations << " "                                             //
      << prefix << "max_degree=" << large_degree_threshold << " "                                         //
      << prefix << "merge_nonadjacent_clusters_threshold=" << merge_nonadjacent_clusters_threshold << " " //
      << prefix << "merge_isolated_clusters=" << merge_isolated_clusters << " "                           //
      << prefix << "max_num_neighbors=" << max_num_neighbors << " ";                                      //
}

void LabelPropagationRefinementContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "num_iterations=" << num_iterations << " "        //
      << prefix << "max_degree=" << large_degree_threshold << " "    //
      << prefix << "max_num_neighbors=" << max_num_neighbors << " "; //
}

void FMRefinementContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "stopping_rule=" << stopping_rule << " "             //
      << prefix << "num_fruitless_moves=" << num_fruitless_moves << " " //
      << prefix << "alpha=" << alpha << " ";                            //
}

void BalancerRefinementContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "timepoint=" << timepoint << " "  //
      << prefix << "graphutils=" << algorithm << " "; //
}

void RefinementContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "graphutils=" << algorithm << " "; //

  lp.print(out, prefix + "lp.");
  fm.print(out, prefix + "fm.");
  balancer.print(out, prefix + "balancer.");
}

void InitialPartitioningContext::print(std::ostream &out, const std::string &prefix) const {
  coarsening.print(out, prefix + "coarsening.");
  refinement.print(out, prefix + "refinement.");
  out << prefix << "mode=" << mode << " "                                                                 //
      << prefix << "repetition_multiplier=" << repetition_multiplier << " "                               //
      << prefix << "min_num_repetitions=" << min_num_repetitions << " "                                   //
      << prefix << "max_num_repetitions=" << max_num_repetitions << " "                                   //
      << prefix << "num_seed_iterations=" << num_seed_iterations << " "                                   //
      << prefix << "use_adaptive_epsilon=" << use_adaptive_epsilon << " "                                 //
      << prefix << "use_adaptive_bipartitioner_selection=" << use_adaptive_bipartitioner_selection << " " //
      << prefix << "multiplier_exponent=" << multiplier_exponent << " "                                   //
      << prefix << "parallelize_bisections=" << parallelize_bisections << " ";                            //
}

void DebugContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "just_sanitize_args=" << just_sanitize_args << " "; //
}

void ParallelContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "use_interleaved_numa_allocation=" << use_interleaved_numa_allocation << " " //
      << prefix << "num_threads=" << num_threads << " ";                                        //
}

void Context::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "graph_filename=" << graph_filename << " "           //
      << prefix << "seed=" << seed << " "                               //
      << prefix << "save_output_partition=" << save_partition << " "    //
      << prefix << "partition_filename=" << partition_filename << " "   //
      << prefix << "partition_directory=" << partition_directory << " " //
      << prefix << "ignore_weights=" << ignore_weights << " "           //
      << prefix << "quiet=" << quiet << " ";                            //

  partition.print(out, prefix + "partition.");
  coarsening.print(out, prefix + "coarsening.");
  initial_partitioning.print(out, prefix + "initial_partitioning.");
  refinement.print(out, prefix + "refinement.");
  debug.print(out, prefix + "debug.");
  parallel.print(out, prefix + "parallel.");
}

void Context::setup(const Graph &graph) {
  partition.setup(graph);
  partition.setup_max_block_weight();
}

Context create_default_context() {
  // clang-format off
  return { // Context
    .graph_filename = "",
    .seed = 0,
    .save_partition = false,
    .partition_directory = "./",
    .partition_filename = "", // generate filename
    .ignore_weights = false,
    .quiet = false,
    .partition = { // Context -> Partition
      .mode = PartitioningMode::DEEP,
      .epsilon = 0.03,
      .k = 2,
      .fast_initial_partitioning = false,
    },
    .coarsening = { // Context -> Coarsening
      .algorithm = ClusteringAlgorithm::LABEL_PROPAGATION,
      .lp = { // Context -> Coarsening -> Label Propagation
        .num_iterations = 5,
        .large_degree_threshold = 1000000,
        .merge_nonadjacent_clusters_threshold = 0.5,
        .merge_isolated_clusters = true,
        .max_num_neighbors = 200000,
      },
      .contraction_limit = 2000,
      .enforce_contraction_limit = false,
      .convergence_threshold = 0.05,
      .cluster_weight_limit = ClusterWeightLimit::EPSILON_BLOCK_WEIGHT,
      .cluster_weight_multiplier = 1.0,
    },
    .initial_partitioning = { // Context -> Initial Partitioning
      .coarsening = { // Context -> Initial Partitioning -> Coarsening
        .algorithm = ClusteringAlgorithm::LABEL_PROPAGATION,
        .lp = { // Context -> Initial Partitioning -> Coarsening -> Label Propagation
          .num_iterations = 1, // no effect
          .large_degree_threshold = 1000000, // no effect
          .merge_nonadjacent_clusters_threshold = 0.5, // no effect
          .merge_isolated_clusters = true, // no effect
          .max_num_neighbors = 200000, // no effect
        },
        .contraction_limit = 20,
        .enforce_contraction_limit = false, // no effect
        .convergence_threshold = 0.05,
        .cluster_weight_limit = ClusterWeightLimit::BLOCK_WEIGHT,
        .cluster_weight_multiplier = 1.0 / 12.0,
      },
      .refinement = { // Context -> Initial Partitioning -> Refinement
        .algorithm = RefinementAlgorithm::TWO_WAY_FM,
        .lp = {},
        .fm = { // Context -> Initial Partitioning -> Refinement -> FM
          .stopping_rule = FMStoppingRule::SIMPLE,
          .num_fruitless_moves = 100,
          .alpha = 1.0,
          .num_iterations = 5,
          .improvement_abortion_threshold = 0.0001,
        },
        .balancer = { // Context -> Initial Partitioning -> Refinement -> Balancer
          .algorithm = BalancingAlgorithm::BLOCK_LEVEL_PARALLEL_BALANCER,
          .timepoint = BalancingTimepoint::BEFORE_KWAY_REFINEMENT,
        },
      },
      .mode = InitialPartitioningMode::SYNCHRONOUS_PARALLEL,
      .repetition_multiplier = 1.0,
      .min_num_repetitions = 10,
      .min_num_non_adaptive_repetitions = 5,
      .max_num_repetitions = 50,
      .num_seed_iterations = 1,
      .use_adaptive_epsilon = true,
      .use_adaptive_bipartitioner_selection = true,
      .multiplier_exponent = 0,
      .parallelize_bisections = false,
    },
    .refinement = { // Context -> Refinement
      .algorithm = RefinementAlgorithm::LABEL_PROPAGATION,
      .lp = { // Context -> Refinement -> Label Propagation
        .num_iterations = 5,
        .large_degree_threshold = 1000000,
        .max_num_neighbors = std::numeric_limits<NodeID>::max(),
      },
      .fm = {},
      .balancer = { // Context -> Refinement -> Balancer
        .algorithm = BalancingAlgorithm::BLOCK_LEVEL_PARALLEL_BALANCER,
        .timepoint = BalancingTimepoint::BEFORE_KWAY_REFINEMENT,
      },
    },
    .debug = { // Context -> Debug
      .just_sanitize_args = false,
    },
    .parallel = { // Context -> Parallel
      .use_interleaved_numa_allocation = true,
      .num_threads = 1,
    },
  };
  // clang-format on
}

Context create_default_context(const Graph &graph, const BlockID k, const double epsilon) {
  Context context = create_default_context();
  context.setup(graph);
  context.partition.k = k;
  context.partition.epsilon = epsilon;
  context.partition.setup_max_block_weight();
  return context;
}

PartitionContext create_bipartition_context(const PartitionContext &k_p_ctx, const Graph &subgraph,
                                            const BlockID final_k1, const BlockID final_k2) {
  PartitionContext two_p_ctx{};
  two_p_ctx.setup(subgraph);
  two_p_ctx.k = 2;
  two_p_ctx.epsilon = compute_2way_adaptive_epsilon(k_p_ctx, subgraph.total_node_weight(), final_k1 + final_k2);
  two_p_ctx.setup_max_block_weight(scalable_vector<BlockID>{final_k1, final_k2});
  return two_p_ctx;
}

std::ostream &operator<<(std::ostream &out, const Context &context) {
  context.print(out);
  return out;
}

double compute_2way_adaptive_epsilon(const PartitionContext &p_ctx, const NodeWeight subgraph_total_node_weight,
                                     const BlockID subgraph_final_k) {
  ASSERT(subgraph_final_k > 1);
  const double base = (1.0 + p_ctx.epsilon) * subgraph_final_k * p_ctx.total_node_weight / p_ctx.k /
                      subgraph_total_node_weight;
  const double exponent = 1.0 / math::ceil_log2(subgraph_final_k);
  const double epsilon_prime = std::pow(base, exponent) - 1.0;
  const double adaptive_epsilon = std::max(epsilon_prime, 0.0001);
  return adaptive_epsilon;
}

NodeWeight compute_max_cluster_weight(const NodeID n, const NodeWeight total_node_weight,
                                      const PartitionContext &input_p_ctx, const CoarseningContext &c_ctx) {
  double max_cluster_weight = 0.0;

  switch (c_ctx.cluster_weight_limit) {
    case ClusterWeightLimit::EPSILON_BLOCK_WEIGHT: {
      const BlockID k_prime = std::clamp<BlockID>(n / c_ctx.contraction_limit, 2, input_p_ctx.k);
      max_cluster_weight = (input_p_ctx.epsilon * total_node_weight) / k_prime;
      break;
    }

    case ClusterWeightLimit::BLOCK_WEIGHT:
      max_cluster_weight = (1.0 + input_p_ctx.epsilon) * total_node_weight / input_p_ctx.k;
      break;

    case ClusterWeightLimit::ONE: max_cluster_weight = 1.0; break;

    case ClusterWeightLimit::ZERO: max_cluster_weight = 0.0; break;
  }

  return static_cast<NodeWeight>(max_cluster_weight * c_ctx.cluster_weight_multiplier);
}

NodeWeight compute_max_cluster_weight(const Graph &c_graph, const PartitionContext &input_p_ctx,
                                      const CoarseningContext &c_ctx) {
  return compute_max_cluster_weight(c_graph.n(), c_graph.total_node_weight(), input_p_ctx, c_ctx);
}
} // namespace kaminpar
