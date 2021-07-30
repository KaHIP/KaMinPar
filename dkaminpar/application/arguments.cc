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
#include "dkaminpar/application/arguments.h"

#include "kaminpar/application/arguments.h"

#include <string>

namespace dkaminpar::app {
using namespace std::string_literals;

void create_coarsening_options(CoarseningContext &c_ctx, kaminpar::Arguments &args, const std::string &name,
                               const std::string &prefix) {
  // clang-format off
  args.group(name, prefix)
      .argument(prefix + "-algorithm", "Coarsening algorithm, possible values: {"s + coarsening_algorithm_names() + "}.", &c_ctx.algorithm, coarsening_algorithm_from_string)
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