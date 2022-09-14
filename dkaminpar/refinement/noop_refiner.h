/*******************************************************************************
 * @file:   noop_refiner.h
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 * @brief:  Refiner that does nothing.
 ******************************************************************************/
#pragma once

#include "dkaminpar/refinement/refiner.h"

namespace kaminpar::dist {
class NoopRefiner : public Refiner {
public:
    void initialize(const DistributedGraph&, const PartitionContext&) override;
    void refine(DistributedPartitionedGraph&) override;
};
} // namespace kaminpar::dist
