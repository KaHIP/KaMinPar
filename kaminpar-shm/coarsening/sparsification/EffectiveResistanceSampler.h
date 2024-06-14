//
// Created by badger on 5/19/24.
//

#pragma once

#include <julia.h>
// JULIA_DEFINE_FAST_TLS // only define this once, in an executable (not in a shared library) if you want fast code.

#include "Sampler.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm::sparsification {
class EffectiveResistanceSampler : public Sampler {
public:
  EffectiveResistanceSampler();
  ~EffectiveResistanceSampler();
  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override;
};
} // namespace kaminpar::shm::sparsification
