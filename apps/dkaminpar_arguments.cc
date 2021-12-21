/*******************************************************************************
 * @file:   arguments.cc
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:
 ******************************************************************************/
#include "apps/dkaminpar_arguments.h"

#include "kaminpar/application/arguments.h"

#include <string>

namespace dkaminpar::app {
using namespace std::string_literals;

#ifdef KAMINPAR_GRAPHGEN
void create_graphgen_options(graphgen::GeneratorContext &g_ctx, kaminpar::Arguments &args, const std::string &name,
                             const std::string &prefix) {
  // clang-format off
  args.group(name, prefix)
      .argument(prefix, "Graph generator, possible values: {" + graphgen::generator_type_names() + "}.", &g_ctx.type, graphgen::generator_type_from_string)
      .argument(prefix + "-n", "Number of nodes in the graph.", &g_ctx.n)
      .argument(prefix + "-m", "Number of edges in the graph.", &g_ctx.m)
      .argument(prefix + "-k", "Number of chunks (depending on model). Can be 0.", &g_ctx.k)
      .argument(prefix + "-d", "Average degree (depending on model).", &g_ctx.d)
      .argument(prefix + "-p", "P?", &g_ctx.p)
      .argument(prefix + "-r", "Radius (depending on model).", &g_ctx.r)
      .argument(prefix + "-gamma", "Power law exponent (depending on model)", &g_ctx.gamma)
      .argument(prefix + "-save-graph", "Write the generated graph to the hard disk.", &g_ctx.save_graph)
      ;
  // clang-format on
}
#endif // KAMINPAR_GRAPHGEN

void create_coarsening_label_propagation_options(LabelPropagationCoarseningContext &lp_ctx, kaminpar::Arguments &args,
                                                 const std::string &name, const std::string &prefix) {
  // clang-format off
  args.group(name, prefix)
      .argument(prefix + "-iterations", "Maximum number of LP iterations.", &lp_ctx.num_iterations)
      .argument(prefix + "-total-num-chunks", "Number of communication chunks times number of PEs.", &lp_ctx.total_num_chunks)
      .argument(prefix + "-min-num-chunks", "Minimum number of communication chunks.", &lp_ctx.min_num_chunks)
      .argument(prefix + "-num-chunks", "Number of communication chunks. If set to 0, the value is computed from total-num-chunks.", &lp_ctx.num_chunks)
      ;
  // clang-format on
}

void create_coarsening_options(CoarseningContext &c_ctx, kaminpar::Arguments &args, const std::string &name,
                               const std::string &prefix) {
  // clang-format off
  args.group(name, prefix)
      .argument(prefix + "-contraction-limit", "Contraction limit", &c_ctx.contraction_limit)
      .argument(prefix + "-use-local-coarsening", "Enable local coarsening before global coarsening.", &c_ctx.use_local_clustering)
      .argument(prefix + "-use-global-coarsening", "Enable global coarsening after local coarsening.", &c_ctx.use_global_clustering)
      .argument(prefix + "-global-clustering-algorithm", "Clustering algorithm, possible values: {"s + global_clustering_algorithm_names() + "}.", &c_ctx.global_clustering_algorithm, global_clustering_algorithm_from_string)
      .argument(prefix + "-global-contraction-algorithm", "Contraction algorithm, possible values: {"s + global_contraction_algorithm_names() + "}.", &c_ctx.global_contraction_algorithm, global_contraction_algorithm_from_string)
      .argument(prefix + "-cluster-weight-limit", "Function to compute the cluster weight limit, possible values: {"s + shm::cluster_weight_limit_names() + "}.", &c_ctx.cluster_weight_limit, shm::cluster_weight_limit_from_string)
      .argument(prefix + "-cluster-weight-multiplier", "Multiplier for the cluster weight limit.", &c_ctx.cluster_weight_multiplier)
      ;
  // clang-format on
  create_coarsening_label_propagation_options(c_ctx.local_lp, args, name + " -> Local Label Propagation",
                                              prefix + "-llp");
  create_coarsening_label_propagation_options(c_ctx.global_lp, args, name + " -> Global Label Propagation",
                                              prefix + "-glp");
}

void create_refinement_label_propagation_options(LabelPropagationRefinementContext &lp_ctx, kaminpar::Arguments &args,
                                                 const std::string &name, const std::string &prefix) {
  // clang-format off
  args.group(name, prefix)
      .argument(prefix + "-iterations", "Maximum number of LP iterations.", &lp_ctx.num_iterations)
      .argument(prefix + "-total-num-chunks", "Number of communication chunks times number of PEs.", &lp_ctx.total_num_chunks)
      .argument(prefix + "-min-num-chunks", "Minimum number of communication chunks.", &lp_ctx.min_num_chunks)
      .argument(prefix + "-num-chunks", "Number of communication chunks. If set to 0, the value is computed from total-num-chunks.", &lp_ctx.num_chunks)
      ;
  // clang-format on
}

void create_refinement_options(RefinementContext &r_ctx, kaminpar::Arguments &args, const std::string &name,
                               const std::string &prefix) {
  // clang-format off
  args.group(name, prefix)
      .argument(prefix + "-algorithm", "Refinement algorithm, possible values: {"s + kway_refinement_algorithm_names() + "}.", &r_ctx.algorithm, kway_refinement_algorithm_from_string)
      ;
  // clang-format on
  create_refinement_label_propagation_options(r_ctx.lp, args, name + " -> Label Propagation", prefix + "-lp");
}

void create_initial_partitioning_options(InitialPartitioningContext &i_ctx, kaminpar::Arguments &args,
                                         const std::string &name, const std::string &prefix) {
  // clang-format off
  args.group(name, prefix)
      .argument(prefix + "-algorithm", "Initial partitioning algorithm, possible values: {"s + initial_partitioning_algorithm_names() + "}.", &i_ctx.algorithm, initial_partitioning_algorithm_from_string)
      ;
  // clang-format on
  shm::app::create_algorithm_options(i_ctx.sequential, args, "Initial Partitioning -> KaMinPar -> ", prefix + "i-");
}

void create_miscellaneous_context_options(Context &ctx, kaminpar::Arguments &args, const std::string &name,
                                          const std::string &prefix) {
  // clang-format off
  args.group(name, prefix)
      .argument("epsilon", "Maximum allowed imbalance.", &ctx.partition.epsilon, 'e')
      .argument("threads", "Maximum number of threads to be used.", &ctx.parallel.num_threads, 't')
      .argument("seed", "Seed for random number generator.", &ctx.seed, 's')
      .argument("quiet", "Do not produce any output to stdout.", &ctx.quiet, 'q')
      .argument("edge-balanced", "Read input graph such that edges are distributed evenly across PEs.", &ctx.load_edge_balanced, 'E')
      ;
  // clang-format on
}

void create_mandatory_options(Context &ctx, kaminpar::Arguments &args, const std::string &name) {
  // clang-format off
  args.group(name, "", true)
      .argument("k", "Number of blocks", &ctx.partition.k, 'k')
      .argument("graph", "Graph to partition", &ctx.graph_filename, 'G')
      ;
  // clang-format on
}

void create_context_options(ApplicationContext &a_ctx, kaminpar::Arguments &args) {
  create_mandatory_options(a_ctx.ctx, args, "Mandatory");
  create_miscellaneous_context_options(a_ctx.ctx, args, "Miscellaneous", "m");
  create_coarsening_options(a_ctx.ctx.coarsening, args, "Coarsening", "c");
  create_initial_partitioning_options(a_ctx.ctx.initial_partitioning, args, "Initial Partitioning", "i");
  create_refinement_options(a_ctx.ctx.refinement, args, "Refinement", "r");
#ifdef KAMINPAR_GRAPHGEN
  create_graphgen_options(a_ctx.generator, args, "Graph Generation", "g");
#endif // KAMINPAR_GRAPHGEN
}

ApplicationContext parse_options(int argc, char *argv[]) {
#ifdef KAMINPAR_GRAPHGEN
  ApplicationContext a_ctx{create_default_context(), {}};
#else  // KAMINPAR_GRAPHGEN
  ApplicationContext a_ctx{create_default_context()};
#endif // KAMINPAR_GRAPHGEN
  kaminpar::Arguments arguments;
  create_context_options(a_ctx, arguments);
  arguments.parse(argc, argv);
  return a_ctx;
}
} // namespace dkaminpar::app