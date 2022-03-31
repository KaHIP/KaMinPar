/*******************************************************************************
 * @file:   noop_refiner.h
 *
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 * @brief:  Refiner that does nothing.
 ******************************************************************************/
#pragma once

#include "dkaminpar/refinement/i_distributed_refiner.h"

namespace dkaminpar {
class NoopRefiner : public IDistributedRefiner {
public:
    void initialize(const DistributedGraph&, const PartitionContext&) override;
    void refine(DistributedPartitionedGraph&) override;
};
} // namespace dkaminpar