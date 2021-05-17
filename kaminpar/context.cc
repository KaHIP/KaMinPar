#include "context.h"

#define DEFINE_ENUM_STRING_CONVERSION(type_name, prefix_name)                                                          \
  struct type_name##Dummy {                                                                                            \
    static std::map<type_name, std::string_view> enum_to_name;                                                         \
  };                                                                                                                   \
                                                                                                                       \
  type_name prefix_name##_from_string(const std::string &searched) {                                                   \
    for (const auto [value, name] : type_name##Dummy::enum_to_name) {                                                  \
      if (name == searched) { return value; }                                                                          \
    }                                                                                                                  \
    throw std::runtime_error("invalid name: "s + searched);                                                            \
  }                                                                                                                    \
                                                                                                                       \
  std::ostream &operator<<(std::ostream &os, const type_name &value) {                                                 \
    return os << type_name##Dummy::enum_to_name.find(value)->second;                                                   \
  }                                                                                                                    \
                                                                                                                       \
  std::string prefix_name##_names(const std::string &sep) {                                                            \
    std::stringstream names;                                                                                           \
    bool first = true;                                                                                                 \
    for (const auto [value, name] : type_name##Dummy::enum_to_name) {                                                  \
      if (!first) { names << sep; }                                                                                    \
      names << name;                                                                                                   \
      first = false;                                                                                                   \
    }                                                                                                                  \
    return names.str();                                                                                                \
  }                                                                                                                    \
                                                                                                                       \
  std::map<type_name, std::string_view> type_name##Dummy::enum_to_name

namespace kaminpar {
using namespace std::string_literals;

//
// Define std::string <-> enum conversion functions
//

DEFINE_ENUM_STRING_CONVERSION(CoarseningAlgorithm, coarsening_algorithm) = {
    {CoarseningAlgorithm::NOOP, "noop"},
    {CoarseningAlgorithm::PARALLEL_LABEL_PROPAGATION, "mt-lp"},
};

DEFINE_ENUM_STRING_CONVERSION(RefinementAlgorithm, refinement_algorithm) = {
    {RefinementAlgorithm::TWO_WAY_FM, "2way-fm"}, //
    {RefinementAlgorithm::PARALLEL_LABEL_PROPAGATION, "lp"},
    {RefinementAlgorithm::NOOP, "noop"}, //
};

DEFINE_ENUM_STRING_CONVERSION(BalancingTimepoint, balancing_timepoint) = {
    {BalancingTimepoint::BEFORE_KWAY_REFINEMENT, "before-kway-ref"},
    {BalancingTimepoint::AFTER_KWAY_REFINEMENT, "after-kway-ref"},
    {BalancingTimepoint::BEFORE_AND_AFTER_KWAY_REFINEMENT, "before-and-after-kway-ref"},
    {BalancingTimepoint::NEVER, "never"},
};

DEFINE_ENUM_STRING_CONVERSION(BalancingAlgorithm, balancing_algorithm) = {
    {BalancingAlgorithm::BLOCK_LEVEL_PARALLEL_BALANCER, "block-level-parallel-balancer"},
};

DEFINE_ENUM_STRING_CONVERSION(FMStoppingRule, fm_stopping_rule) = {
    {FMStoppingRule::SIMPLE, "simple"},
    {FMStoppingRule::ADAPTIVE, "adaptive"},
};

DEFINE_ENUM_STRING_CONVERSION(PartitioningMode, partitioning_mode) = {
    {PartitioningMode::DEEP, "deep"},
    {PartitioningMode::RB, "rb"},
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
      << prefix << "remove_isolated_nodes=" << remove_isolated_nodes << " "          //
      << prefix << "fast_initial_partitioning=" << fast_initial_partitioning << " "; //
}

void CoarseningContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "enable=" << enable << " "                                                                       //
      << prefix << "algorithm=" << algorithm << " "                                                                 //
      << prefix << "contraction_limit=" << contraction_limit << " "                                                 //
      << prefix << "use_adaptive_contraction_limit=" << use_adaptive_contraction_limit << " "                       //
      << prefix << "min_shrink_factor=" << min_shrink_factor << " "                                                 //
      << prefix << "adaptive_cluster_weight_multiplier=" << adaptive_cluster_weight_multiplier << " "               //
      << prefix << "use_adaptive_epsilon=" << use_adaptive_epsilon << " "                                           //
      << prefix << "block_based_cluster_weight_factor=" << block_based_cluster_weight_factor << " "                 //
      << prefix << "num_iterations=" << num_iterations << " "                                                       //
      << prefix << "large_degree_threshold=" << large_degree_threshold << " "                                       //
      << prefix << "shrink_factor_abortion_threshold=" << shrink_factor_abortion_threshold << " "                   //
      << prefix << "nonadjacent_clustering_fraction_threshold=" << nonadjacent_clustering_fraction_threshold << " " //
      << prefix << "randomize_chunk_order=" << randomize_chunk_order << " "                                         //
      << prefix << "merge_singleton_clusters=" << merge_singleton_clusters << " ";                                  //
}

void LabelPropagationRefinementContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "num_iterations=" << num_iterations << " "                 //
      << prefix << "large_degree_threshold=" << large_degree_threshold << " " //
      << prefix << "randomize_chunk_order=" << randomize_chunk_order << " ";  //
}

void FMRefinementContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "stopping_rule=" << stopping_rule << " "             //
      << prefix << "num_fruitless_moves=" << num_fruitless_moves << " " //
      << prefix << "alpha=" << alpha << " ";                            //
}

void BalancerRefinementContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "timepoint=" << timepoint << " "  //
      << prefix << "algorithm=" << algorithm << " "; //
}

void RefinementContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "algorithm=" << algorithm << " "; //

  lp.print(out, prefix + "lp.");
  fm.print(out, prefix + "fm.");
  balancer.print(out, prefix + "balancer.");
}

void InitialPartitioningContext::print(std::ostream &out, const std::string &prefix) const {
  coarsening.print(out, prefix + "coarsening.");
  refinement.print(out, prefix + "refinement.");
  out << prefix << "repetition_multiplier=" << repetition_multiplier << " "                               //
      << prefix << "min_num_repetitions=" << min_num_repetitions << " "                                   //
      << prefix << "max_num_repetitions=" << max_num_repetitions << " "                                   //
      << prefix << "num_seed_iterations=" << num_seed_iterations << " "                                   //
      << prefix << "use_adaptive_epsilon=" << use_adaptive_epsilon << " "                                 //
      << prefix << "use_adaptive_bipartitioner_selection=" << use_adaptive_bipartitioner_selection << " " //
      << prefix << "parallelize=" << parallelize << " "                                                   //
      << prefix << "parallelize_synchronized=" << parallelize_synchronized << " "                         //
      << prefix << "multiplier_exponent=" << multiplier_exponent << " ";                                  //
}

void DebugContext::print(std::ostream &out, const std::string &prefix) const {
  out << prefix << "just_sanitize_args=" << just_sanitize_args << " " //
      << prefix << "force_clean_build=" << force_clean_build << " ";  //
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
      << prefix << "show_local_timers=" << show_local_timers << " ";    //

  partition.print(out, "partition.");
  coarsening.print(out, "coarsening.");
  initial_partitioning.print(out, "initial_partitioning.");
  refinement.print(out, "refinement.");
  debug.print(out, "debug.");
  parallel.print(out, "parallel.");
}

void Context::setup(const Graph &graph) {
  partition.setup(graph);
  partition.setup_max_block_weight();
  if (coarsening.use_adaptive_contraction_limit) {
    coarsening.contraction_limit = std::min(coarsening.contraction_limit, graph.n() / partition.k);
  }
}

Context Context::create_default() {
  Context context;
  context.seed = 0;

  // partition
  context.partition.epsilon = 0.03;

  // coarsening
  context.coarsening.algorithm = CoarseningAlgorithm::PARALLEL_LABEL_PROPAGATION;
  context.coarsening.contraction_limit = 2000;
  context.coarsening.num_iterations = 5;
  context.coarsening.shrink_factor_abortion_threshold = 0.05;

  // initial partitioning -> coarsening
  context.initial_partitioning.coarsening.algorithm = CoarseningAlgorithm::PARALLEL_LABEL_PROPAGATION;
  context.initial_partitioning.coarsening.block_based_cluster_weight_factor = 12.0;
  context.initial_partitioning.coarsening.contraction_limit = 20;
  context.initial_partitioning.coarsening.num_iterations = 10; // unused? initial coarsener performs only one round
  context.initial_partitioning.coarsening.shrink_factor_abortion_threshold = 0.05;

  // initial partitioning -> refinement
  context.initial_partitioning.refinement.algorithm = RefinementAlgorithm::TWO_WAY_FM;
  context.initial_partitioning.refinement.fm.stopping_rule = FMStoppingRule::SIMPLE;
  context.initial_partitioning.refinement.fm.num_iterations = 5;

  // refinement
  context.refinement.algorithm = RefinementAlgorithm::PARALLEL_LABEL_PROPAGATION;
  context.refinement.lp.num_iterations = 5;
  context.refinement.lp.large_degree_threshold = 1000000;

  return context;
}

Context Context::create_default_for(const Graph &graph, const BlockID k, const double epsilon) {
  Context context = create_default();
  context.setup(graph);
  context.partition.k = k;
  context.partition.epsilon = epsilon;
  context.partition.setup_max_block_weight();
  return context;
}

PartitionContext Context::create_bipartition_partition_context(const Graph &subgraph, const BlockID final_k1,
                                                               const BlockID final_k2) const {
  PartitionContext p_ctx{};
  p_ctx.setup(subgraph);
  p_ctx.k = 2;
  p_ctx.epsilon = compute_2way_adaptive_epsilon(partition, subgraph.total_node_weight(), final_k1 + final_k2);
  p_ctx.setup_max_block_weight(scalable_vector<BlockID>{final_k1, final_k2});
  return p_ctx;
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

NodeWeight compute_max_cluster_weight(const Graph &c_graph, const PartitionContext &input_p_ctx,
                                      const CoarseningContext &c_ctx, const CoarseningContext &input_c_ctx) {
  // Adaptive cluster weight as described in the paper
  const BlockID k_prime = std::clamp<BlockID>(c_graph.n() / c_ctx.contraction_limit, 2, input_p_ctx.k);
  const double eps = c_ctx.use_adaptive_epsilon
                         ? compute_2way_adaptive_epsilon(input_p_ctx, c_graph.total_node_weight(), k_prime)
                         : input_p_ctx.epsilon;
  const NodeWeight adaptive_limit = c_ctx.adaptive_cluster_weight_multiplier * (eps * c_graph.total_node_weight()) /
                                    k_prime;

  // Cluster weight based on the maximum block weight in the partitioned graph
  // block_based_cluster_weight_factor
  const double max_block_weight = (1.0 + eps) * c_graph.total_node_weight() / input_p_ctx.k; // during rb, this is 2
  const NodeWeight block_based_limit = max_block_weight / c_ctx.block_based_cluster_weight_factor;

  // Cluster weight based on average node weight: if all nodes have the same weight, this allows the graph to shrink
  // by up to factor coarsening.min_shrink_factor
  // Default: ignored (min_shrink_factor set to 0)
  const NodeWeight shrink_based_limit = std::min<NodeWeight>(c_ctx.min_shrink_factor * c_graph.total_node_weight() /
                                                                 std::min<NodeID>(c_graph.n(),
                                                                                  2 * input_c_ctx.contraction_limit),
                                                             c_graph.total_node_weight() - 1);

  return std::max<NodeWeight>({adaptive_limit, block_based_limit, shrink_based_limit});
}
} // namespace kaminpar
