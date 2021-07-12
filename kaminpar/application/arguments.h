/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#pragma once

#include "kaminpar/application/arguments_parser.h"
#include "kaminpar/context.h"

namespace kaminpar::app {
// clang-format off
void create_coarsening_context_options(CoarseningContext &c_ctx, Arguments &args, const std::string &name, const std::string &prefix) {
  using namespace std::string_literals;
  args.group(name, prefix)
      .argument(prefix + "-algorithm", "Coarsening algorithm, possible values: {" + clustering_algorithm_names() + "}.", &c_ctx.algorithm, clustering_algorithm_from_string)
        .line("- noop: assign each node to its own cluster, effectively disabling clustering")
        .line("- lp: compute clustering using parallel label propagation")
      .argument(prefix + "-contraction-limit", "Ideally, we always perform a bisection on a graph of size 2 * C.", &c_ctx.contraction_limit, 'C')
      .argument(prefix + "-enforce-contraction-limit", "If set, enforce the ideal coarsening situation.", &c_ctx.enforce_contraction_limit)
      .argument(prefix + "-convergence-threshold", "Coarsening converged once the size of the graph shrunk by less than this factor.", &c_ctx.convergence_threshold)
      .argument(prefix + "-cluster-weight-limit", "Rule used to compute the maximum cluster weight, possible values: {" + cluster_weight_limit_names() + "}.", &c_ctx.cluster_weight_limit, cluster_weight_limit_from_string)
        .line("The limit can be multiplied by any constant using the --cluster_weight_multiplier argument.")
        .line("- epsilon-block-weight: c_max = eps * min{n' / C, k} when coarsening a graph with n' nodes.")
        .line("- static_block_weight: c_max = n / k")
        .line("- one: c_max = 1")
        .line("- zero: c_max = 0")
      .argument(prefix + "-cluster_weight_multiplier", "Multiplier for the maximum cluster weight limit.", &c_ctx.cluster_weight_multiplier)
      ;
}
// clang-format on

// clang-format off
void create_coarsening_lp_context_options(LabelPropagationCoarseningContext &c_lp_ctx, Arguments &args, const std::string &name, const std::string &prefix) {
  args.group(name, prefix)
      .argument(prefix + "-num-iterations", "Number of label propagation iterations.", &c_lp_ctx.num_iterations)
      .argument(prefix + "-large-degree-threshold", "Ignore all nodes with degree higher than this during coarsening.", &c_lp_ctx.large_degree_threshold)
      .argument(prefix + "-merge-nonadjacent-clusters-threshold", "If the graph shrunk by less than this factor, consider merging nonadjacent singleton clusters.", &c_lp_ctx.merge_nonadjacent_clusters_threshold)
      .argument(prefix + "-merge-singleton-clusters", "If set, cluster isolated nodes.", &c_lp_ctx.merge_singleton_clusters)
      .argument(prefix + "-max-num-neighbors", "Maximum numbers of neighbors that are scanned before deciding a nodes best cluster.", &c_lp_ctx.max_num_neighbors)
      ;
}
// clang-format on

// clang-format off
void create_mandatory_context_options(Context &ctx, Arguments &args, const std::string &name) {
  args.group(name, "", true)
      .argument("k", "Number of blocks", &ctx.partition.k, 'k')
      .argument("graph", "Graph to partition", &ctx.graph_filename, 'G')
      ;
}
// clang-format on

// clang-format off
void create_miscellaneous_context_options(Context &ctx, Arguments &args, const std::string &name, const std::string &prefix) {
  args.group(name, prefix)
      .argument("epsilon", "Maximum allowed imbalance.", &ctx.partition.epsilon, 'e')
      .argument("threads", "Maximum number of threads to be used.", &ctx.parallel.num_threads, 't')
      .argument("seed", "Seed for random number generator.", &ctx.seed, 's')
      .argument("save-partition", "Save the partition to a file.", &ctx.save_partition)
      .argument("partition-directory", "[--save-partition] Directory for the partition file.", &ctx.partition_directory)
      .argument("partition-name", "[--save-partition] Filename for the partition file. If empty, one is generated.", &ctx.partition_filename)
      .argument("use-interleaved-numa-allocation", "Interleave allocations across NUMA nodes round-robin style.", &ctx.parallel.use_interleaved_numa_allocation)
      .argument("fast-ip", "Use cheaper initial partitioning if k is larger than this.", &ctx.partition.fast_initial_partitioning)
      .argument("mode", "Partitioning mode, possible values: {" + partitioning_mode_names() + "}.", &ctx.partition.mode, partitioning_mode_from_string)
      .argument("quiet", "Do not produce any output.", &ctx.quiet, 'q')
      ;
}
// clang-format on

// clang-format off
void create_debug_context_options(DebugContext &d_ctx, Arguments &args, const std::string &name, const std::string &prefix) {
  args.group(name, prefix)
      .argument(prefix + "-just-sanitize-args", "Sanitize command line arguments and exit.", &d_ctx.just_sanitize_args)
      .argument(prefix + "-force-clean-build", "Abort execution if working directory is dirty.", &d_ctx.force_clean_build)
      ;
}
// clang-format on

// clang-format off
void create_initial_partitioning_context_options(InitialPartitioningContext &i_ctx, Arguments &args, const std::string &name, const std::string &prefix) {
  args.group(name, prefix)
      .argument(prefix + "-mode", "Initial partitioning mode, possible values are: {" + initial_partitioning_mode_names() + "}.", &i_ctx.mode, initial_partitioning_mode_from_string)
        .line("- sequential: do not diversify initial partitioning by creating multiple copies of the original graph")
        .line("- async-parallel: create diversified copies of coarser graphs and process them asynchronously")
        .line("- sync-parallel: create diversified copies of coarser graphs and process them in lock step")
      .argument(prefix + "-multiplier-exp", "", &i_ctx.multiplier_exponent)
      .argument(prefix + "-rep-multiplier", "Multiplier for the number of attempts at computing an initial bisection.", &i_ctx.repetition_multiplier)
      .argument(prefix + "-min-repetitions", "Minimum number of attempts at computing an initial bisection (per bipartition algorithm). A bipartitioning algorithm might be invoked less than specified if it is unlikely to find the best cut.", &i_ctx.min_num_repetitions)
      .argument(prefix + "-min-non-adaptive-repetitions", "Minimum number of attempts at computing an initial bisection (per bipartition algorithm) before excluding bipartitioning algorithms unlikely to find the best cut.", &i_ctx.min_num_non_adaptive_repetitions)
      .argument(prefix + "-max-repetitions", "Maximum number of attempts at computing an initial bisection (per bipartition algorithm).", &i_ctx.max_num_repetitions)
      .argument(prefix + "-num-seed-iterations", "Number of attempts at finding good seed nodes (BFS-based bipartition algorithms).", &i_ctx.num_seed_iterations)
      .argument(prefix + "-use-adaptive-epsilon", "If set, use adaptive epsilon for max block weights during IP.", &i_ctx.use_adaptive_epsilon)
      .argument(prefix + "-use-adaptive-bipartitioner-selection", "If set, determine which bipartitioning algorithms are unlikely to produce good results and run them less often than other algorithms.", &i_ctx.use_adaptive_bipartitioner_selection)
      .argument(prefix + "-parallelize-bisections", "Compute bisections in parallel.", &i_ctx.parallelize_bisections)
      ;
}
// clang-format on

// clang-format off
void create_refinement_context_options(RefinementContext &r_ctx, Arguments &args, const std::string &name, const std::string &prefix) {
  using namespace std::string_literals;

  args.group(name, prefix)
      .argument(prefix + "-algorithm", "Refinement algorithm to be used, possible values: {"s + refinement_algorithm_names() + "}.", &r_ctx.algorithm, refinement_algorithm_from_string)
      ;
}
// clang-format on

// clang-format off
void create_fm_refinement_context_options(FMRefinementContext &fm_ctx, Arguments &args, const std::string &name, const std::string &prefix) {
  using namespace std::string_literals;
  args.group(name, prefix)
      .argument(prefix + "-stopping-rule", "Rule used to determine when to stop the FM algorithm, possible values: {"s + fm_stopping_rule_names() + "}.", &fm_ctx.stopping_rule, fm_stopping_rule_from_string)
      .argument(prefix + "-num-fruitless-moves", "[Simple stopping rule] Number of fruitless moves after which search is aborted.", &fm_ctx.num_fruitless_moves)
      .argument(prefix + "-alpha", "[Adaptive stopping rule] Alpha.", &fm_ctx.alpha)
      .argument(prefix + "-iterations", "Maximum number of iterations.", &fm_ctx.num_iterations)
      .argument(prefix + "-abortion-threshold", "No more FM iterations if the improvement in edge cut of the last iteration was below this threshold.", &fm_ctx.improvement_abortion_threshold)
      ;
}
// clang-format on

// clang-format off
void create_lp_refinement_context_options(LabelPropagationRefinementContext &lp_ctx, Arguments &args, const std::string &name, const std::string &prefix) {
  args.group(name, prefix)
      .argument(prefix + "-num-iterations", "Number label propagation iterations.", &lp_ctx.num_iterations)
      .argument(prefix + "-large-degree-threshold", "Ignore all nodes with degree higher than this.", &lp_ctx.large_degree_threshold)
      .argument(prefix + "-max-num-neighbors", "Maximum number of neighbors to scan before deciding to which block a node is moved.", &lp_ctx.max_num_neighbors)
      ;
}
// clang-format on

// clang-format off
void create_balancer_refinement_context_options(BalancerRefinementContext &b_ctx, Arguments &args, const std::string &name, const std::string &prefix) {
  using namespace std::string_literals;
  args.group(name, prefix)
      .argument(prefix + "-algorithm", "Balancer algorithm, possible values: {"s + balancing_algorithm_names() + "}.", &b_ctx.algorithm, balancing_algorithm_from_string)
      .argument(prefix + "-timepoint", "When do we run the balancer, possible values: {"s + balancing_timepoint_names() + "}.", &b_ctx.timepoint, balancing_timepoint_from_string)
      ;
}
// clang-format on

// clang-format off
void create_context_options(Context &ctx, Arguments &args) {
  create_mandatory_context_options(ctx, args, "Mandatory");
  create_miscellaneous_context_options(ctx, args, "Miscellaneous", "m");
  create_debug_context_options(ctx.debug, args, "Debug", "d");
  create_coarsening_context_options(ctx.coarsening, args, "Coarsening", "c");
  create_coarsening_lp_context_options(ctx.coarsening.lp, args, "Coarsening -> Label Propagation", "c-lp");
  create_coarsening_context_options(ctx.initial_partitioning.coarsening, args, "Initial Partitioning -> Coarsening", "i-c");
  create_coarsening_lp_context_options(ctx.initial_partitioning.coarsening.lp, args, "Coarsening -> Initial Partitioning -> Label Propagation", "i-c-lp");
  create_initial_partitioning_context_options(ctx.initial_partitioning, args, "Initial Partitioning", "i");
  create_refinement_context_options(ctx.initial_partitioning.refinement, args, "Initial Partitioning -> Refinement", "i-r");
  create_fm_refinement_context_options(ctx.initial_partitioning.refinement.fm, args, "Initial Partitioning -> Refinement -> FM", "i-r-fm");
  create_refinement_context_options(ctx.refinement, args, "Refinement", "r");
  create_lp_refinement_context_options(ctx.refinement.lp, args, "Refinement -> Label Propagation", "r-lp");
  create_balancer_refinement_context_options(ctx.refinement.balancer, args, "Refinement -> Balancer", "r-b");
}
// clang-format on

Context parse_options(int argc, char *argv[]) {
  using namespace std::string_literals;
  using namespace std::chrono;

  Context context = create_default_context();

  Arguments arguments;
  create_context_options(context, arguments);
  arguments.parse(argc, argv);
  return context;
}
} // namespace kaminpar::app