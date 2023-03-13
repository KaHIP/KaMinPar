/*******************************************************************************
 * @file:   arguments.h
 * @author: Daniel Seemaier
 * @date:   14.10.2022
 ******************************************************************************/
// clang-format off
#include "common/CLI11.h"
// clang-format on

#include "kaminpar/context.h"

namespace kaminpar::shm {
void create_all_options(CLI::App *app, Context &ctx);
CLI::Option_group *create_partitioning_options(CLI::App *app, Context &ctx);
CLI::Option_group *create_coarsening_options(CLI::App *app, Context &ctx);
CLI::Option_group *create_lp_coarsening_options(CLI::App *app, Context &ctx);
CLI::Option_group *create_initial_partitioning_options(CLI::App *app,
                                                       Context &ctx);
CLI::Option_group *create_initial_refinement_options(CLI::App *app,
                                                     Context &ctx);
CLI::Option_group *create_initial_fm_refinement_options(CLI::App *app,
                                                        Context &ctx);
CLI::Option_group *create_refinement_options(CLI::App *app, Context &ctx);
CLI::Option_group *create_lp_refinement_options(CLI::App *app, Context &ctx);
CLI::Option_group *create_balancer_options(CLI::App *app, Context &ctx);
} // namespace kaminpar::shm
