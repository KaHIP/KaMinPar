/*******************************************************************************
 * @file:   arguments.h
 *
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"

#include "kaminpar/application/arguments_parser.h"
#ifdef KAMINPAR_ENABLE_GRAPHGEN
    #include "apps/dkaminpar_graphgen.h"
#endif // KAMINPAR_ENABLE_GRAPHGEN

namespace dkaminpar::app {
struct ApplicationContext {
    Context ctx;
#ifdef KAMINPAR_ENABLE_GRAPHGEN
    graphgen::GeneratorContext generator;
#endif // KAMINPAR_ENABLE_GRAPHGEN
};

#ifdef KAMINPAR_ENABLE_GRAPHGEN
void create_graphgen_options(
    graphgen::GeneratorContext& g_ctx, kaminpar::Arguments& args, const std::string& name, const std::string& prefix);
#endif // KAMINPAR_ENABLE_GRAPHGEN

void create_coarsening_options(
    CoarseningContext& c_ctx, kaminpar::Arguments& args, const std::string& name, const std::string& prefix);

void create_balancing_options(
    BalancingContext& b_ctx, kaminpar::Arguments& args, const std::string& name, const std::string& prefix);

void create_refinement_options(
    RefinementContext& r_ctx, kaminpar::Arguments& args, const std::string& name, const std::string& prefix);

void create_initial_partitioning_options(
    InitialPartitioningContext& i_ctx, kaminpar::Arguments& args, const std::string& name, const std::string& prefix);

void create_miscellaneous_context_options(
    Context& ctx, kaminpar::Arguments& args, const std::string& name, const std::string& prefix);

void create_mandatory_options(Context& ctx, kaminpar::Arguments& args, const std::string& name);

void create_debug_options(
    DebugContext& d_ctx, kaminpar::Arguments& args, const std::string& name, const std::string& prefix);

void create_context_options(ApplicationContext& ctx, kaminpar::Arguments& args);

ApplicationContext parse_options(int argc, char* argv[]);
} // namespace dkaminpar::app
