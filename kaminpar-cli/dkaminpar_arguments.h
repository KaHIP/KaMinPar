/*******************************************************************************
 * Command line arguments for the distributed partitioner.
 *
 * @file:   dkaminpar_arguments.h
 * @author: Daniel Seemaier
 * @date:   15.10.2022
 ******************************************************************************/
#pragma once

// clang-format off
#include "kaminpar-cli/CLI11.h"
// clang-format on

#include "kaminpar-dist/context.h"

namespace kaminpar::dist {
void create_all_options(CLI::App *app, Context &ctx);

CLI::Option_group *create_partitioning_options(CLI::App *app, Context &ctx);

CLI::Option_group *create_debug_options(CLI::App *app, Context &ctx);

CLI::Option_group *create_initial_partitioning_options(CLI::App *app, Context &ctx);

CLI::Option_group *create_refinement_options(CLI::App *app, Context &ctx);

CLI::Option_group *create_lp_refinement_options(CLI::App *app, Context &ctx);

CLI::Option_group *create_colored_lp_refinement_options(CLI::App *app, Context &ctx);

CLI::Option_group *create_node_balancer_options(CLI::App *app, Context &ctx);

CLI::Option_group *create_cluster_balancer_options(CLI::App *app, Context &ctx);

CLI::Option_group *create_jet_refinement_options(CLI::App *app, Context &ctx);

CLI::Option_group *create_mtkahypar_refinement_options(CLI::App *app, Context &ctx);

CLI::Option_group *create_coarsening_options(CLI::App *app, Context &ctx);

CLI::Option_group *create_local_lp_coarsening_options(CLI::App *app, Context &ctx);

CLI::Option_group *create_global_lp_coarsening_options(CLI::App *app, Context &ctx);

CLI::Option_group *create_hem_coarsening_options(CLI::App *app, Context &ctx);
} // namespace kaminpar::dist
