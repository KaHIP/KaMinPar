/*******************************************************************************
 * @file:   arguments.h
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:
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