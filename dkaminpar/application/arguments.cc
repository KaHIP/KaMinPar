/*******************************************************************************
 * @file:   arguments.cc
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:
 ******************************************************************************/
#include "dkaminpar/application/arguments.h"

#include "kaminpar/application/arguments.h"

#include <string>

namespace dkaminpar::app {
using namespace std::string_literals;

void create_coarsening_label_propagation_options(LabelPropagationCoarseningContext &lp_ctx, kaminpar::Arguments &args,
                                                 const std::string &name, const std::string &prefix) {
  // clang-format off
  args.group(name, prefix)
      .argument(prefix + "-iterations", "Maximum number of LP iterations.", &lp_ctx.num_iterations)
      .argument(prefix + "-total-num-chunks", "Number of communication chunks times number of PEs.", &lp_ctx.total_num_chunks)
      .argument(prefix + "-min-num-chunks", "Minimuim number of communication chunks.", &lp_ctx.min_num_chunks)
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
      ;
  // clang-format on
  create_coarsening_label_propagation_options(c_ctx.local_lp, args, name + " -> Local Label Propagation", prefix + "-llp");
  create_coarsening_label_propagation_options(c_ctx.global_lp, args, name + " -> Global Label Propagation", prefix + "-glp");
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

void create_context_options(Context &ctx, kaminpar::Arguments &args) {
  create_mandatory_options(ctx, args, "Mandatory");
  create_miscellaneous_context_options(ctx, args, "Miscellaneous", "m");
  create_coarsening_options(ctx.coarsening, args, "Coarsening", "c");
  create_initial_partitioning_options(ctx.initial_partitioning, args, "Initial Partitioning", "i");
  create_refinement_options(ctx.refinement, args, "Refinement", "r");
}

Context parse_options(int argc, char *argv[]) {
  Context context = create_default_context();
  kaminpar::Arguments arguments;
  create_context_options(context, arguments);
  arguments.parse(argc, argv);
  return context;
}
} // namespace dkaminpar::app