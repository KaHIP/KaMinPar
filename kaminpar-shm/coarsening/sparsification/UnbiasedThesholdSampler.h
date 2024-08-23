#pragma once
#include "Sampler.h"

namespace kaminpar::shm::sparsification {
class UnbiasedThesholdSampler : public Sampler{
public:
  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override;
private:
  double find_threshold(const CSRGraph &g, EdgeID target_edge_amount);

};
}