/*******************************************************************************
 * Internal constrained k-way FM refinement core factory.
 *
 * @file:   fm_refiner_core.h
 * @author: Daniel Seemaier
 * @date:   20.05.2026
 ******************************************************************************/
#pragma once

#include <memory>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/refinement/refiner.h"

namespace kaminpar::shm::fm {

std::unique_ptr<Refiner> create_fm_core(const Context &ctx, const PartitionedGraph &p_graph);

} // namespace kaminpar::shm::fm
