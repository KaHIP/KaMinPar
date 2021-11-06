/*******************************************************************************
 * @file:   factories.h
 *
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 * @brief:  Instantiates the configured partitioning components.
 ******************************************************************************/
#pragma once

#include "dkaminpar/distributed_context.h"
#include "dkaminpar/initial_partitioning/i_initial_partitioner.h"
#include "dkaminpar/refinement/i_distributed_refiner.h"
#include "dkaminpar/coarsening/i_clustering.h"
#include <memory>

namespace dkaminpar::factory {
std::unique_ptr<IInitialPartitioner> create_initial_partitioner(const Context &ctx);
std::unique_ptr<IDistributedRefiner> create_distributed_refiner(const Context &ctx);
std::unique_ptr<IClustering<GlobalNodeID>> create_global_clustering(const Context &ctx);
}
