#pragma once

#include "kaminpar-shm/coarsening/sparsification/sampler.h"

namespace kaminpar::shm::sparsification {

class UnbiasedThesholdSampler : public Sampler {
public:
  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override;

private:
  double find_threshold(const CSRGraph &g, EdgeID target_edge_amount);
};

} // namespace kaminpar::shm::sparsification
