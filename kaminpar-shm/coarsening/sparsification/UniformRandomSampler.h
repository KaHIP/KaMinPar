//
// Created by badger on 5/6/24.
//

#ifndef UNIFORMRANDOMSAMPLER_H
#define UNIFORMRANDOMSAMPLER_H
#include "Sampler.h"

namespace kaminpar::shm::sparsification {
class UniformRandomSampler : public Sampler {
public:
  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override;
};
} // namespace kaminpar::shm::sparsification

#endif // UNIFORMRANDOMSAMPLER_H
