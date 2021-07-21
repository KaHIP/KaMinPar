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

#include "dkaminpar/distributed_context.h"
#include "kaminpar/application/arguments_parser.h"

namespace dkaminpar::app {
void create_miscellaneous_context_options(DContext &ctx, kaminpar::Arguments &args, const std::string &name, const std::string &prefix) {
  // clang-format off
  args.group(name, prefix)
      .argument("epsilon", "Maximum allowed imbalance.", &ctx.partition.epsilon, 'e')
      .argument("threads", "Maximum number of threads to be used.", &ctx.parallel.num_threads, 't')
      .argument("seed", "Seed for random number generator.", &ctx.seed, 's')
      ;
  // clang-format on
}

void create_mandatory_options(DContext &ctx, kaminpar::Arguments &args, const std::string &name) {
  // clang-format off
  args.group(name, "", true)
    .argument("k", "Number of blocks", &ctx.partition.k, 'k')
    .argument("graph", "Graph to partition", &ctx.graph_filename, 'G')
    ;
  // clang-format on
}

void create_context_options(DContext &ctx, kaminpar::Arguments &args) {
  create_mandatory_options(ctx, args, "Mandatory");
  create_miscellaneous_context_options(ctx, args, "Miscellaneous", "m");
}

DContext parse_options(int argc, char *argv[]) {
  DContext context = create_default_context();
  kaminpar::Arguments arguments;
  create_context_options(context, arguments);
  arguments.parse(argc, argv);
  return context;
}
}