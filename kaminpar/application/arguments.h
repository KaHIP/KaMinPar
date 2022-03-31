/*******************************************************************************
 * @file:   arguments.h
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Command line arguments for KaMinPar. This is part of the core
 * library since we also use it to configure KaMinPar when using the library
 * interface.
 ******************************************************************************/
#pragma once

#include "kaminpar/application/arguments_parser.h"
#include "kaminpar/context.h"

namespace kaminpar::app {
void create_coarsening_context_options(
    CoarseningContext& c_ctx, Arguments& args, const std::string& name, const std::string& prefix);

void create_lp_coarsening_context_options(
    LabelPropagationCoarseningContext& c_lp_ctx, Arguments& args, const std::string& name, const std::string& prefix);

void create_mandatory_context_options(Context& ctx, Arguments& args, const std::string& name);

void create_parallel_context_options(Context& ctx, Arguments& args, const std::string& name, const std::string& prefix);

void create_miscellaneous_context_options(
    Context& ctx, Arguments& args, const std::string& name, const std::string& prefix);

void create_debug_context_options(
    DebugContext& d_ctx, Arguments& args, const std::string& name, const std::string& prefix);

void create_initial_partitioning_context_options(
    InitialPartitioningContext& i_ctx, Arguments& args, const std::string& name, const std::string& prefix);

void create_refinement_context_options(
    RefinementContext& r_ctx, Arguments& args, const std::string& name, const std::string& prefix);

void create_fm_refinement_context_options(
    FMRefinementContext& fm_ctx, Arguments& args, const std::string& name, const std::string& prefix);

void create_lp_refinement_context_options(
    LabelPropagationRefinementContext& lp_ctx, Arguments& args, const std::string& name, const std::string& prefix);

void create_balancer_refinement_context_options(
    BalancerRefinementContext& b_ctx, Arguments& args, const std::string& name, const std::string& prefix);

void create_algorithm_options(
    Context& ctx, Arguments& args, const std::string& global_name_prefix = "", const std::string& global_prefix = "");

void create_context_options(Context& ctx, Arguments& args);

Context parse_options(int argc, char* argv[]);
} // namespace kaminpar::app