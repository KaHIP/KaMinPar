//
// Created by badger on 5/6/24.
//

#ifndef UNIFORMRANDOMSAMPLER_H
#define UNIFORMRANDOMSAMPLER_H
#include "Sampler.h"

namespace kaminpar::shm::sparsification {
class UniformRandomSampler : public Sampler {
private:
  float _probability;

public:
  UniformRandomSampler(float probability) : _probability(probability) {}
  StaticArray<EdgeWeight> sample(const CSRGraph &g) override;
};
} // namespace kaminpar::shm::sparsification

#endif // UNIFORMRANDOMSAMPLER_H
