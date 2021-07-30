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
void create_coarsening_options(CoarseningContext &c_ctx, kaminpar::Arguments &args, const std::string &name,
                               const std::string &prefix);

void create_refinement_options(RefinementContext &r_ctx, kaminpar::Arguments &args, const std::string &name,
                               const std::string &prefix);

void create_initial_partitioning_options(InitialPartitioningContext &i_ctx, kaminpar::Arguments &args,
                                         const std::string &name, const std::string &prefix);

void create_miscellaneous_context_options(Context &ctx, kaminpar::Arguments &args, const std::string &name,
                                          const std::string &prefix);

void create_mandatory_options(Context &ctx, kaminpar::Arguments &args, const std::string &name);

void create_context_options(Context &ctx, kaminpar::Arguments &args);

Context parse_options(int argc, char *argv[]);
} // namespace dkaminpar::app