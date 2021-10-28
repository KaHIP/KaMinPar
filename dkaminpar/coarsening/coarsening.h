/*******************************************************************************
 * @file:   coarsening.h
 *
 * @author: Daniel Seemaier
 * @date:   28.10.2021
 * @brief:  Definitions for coarsening.
 ******************************************************************************/
#pragma once

#include "dkaminpar/distributed_definitions.h"
#include "kaminpar/parallel.h"

namespace dkaminpar::coarsening {
using LocalClustering = scalable_vector<shm::parallel::IntegralAtomicWrapper<NodeID>>;
using GlobalClustering = scalable_vector<shm::parallel::IntegralAtomicWrapper<GlobalNodeID>>;
using LocalMapping = scalable_vector<NodeID>;
using GlobalMapping = scalable_vector<GlobalNodeID>;
}