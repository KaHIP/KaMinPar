//
// Created by badger on 5/8/24.
//

#ifndef FORSTFIRESAMPLER_H
#define FORSTFIRESAMPLER_H
#include "Sampler.h"

namespace kaminpar {
namespace shm {
namespace sparsification {

class ForestFireSampler : public Sampler{
private:
  float _pf;
  float _targetBurntRatio;
public:
  ForestFireSampler(float pf, float targetBurntRatio): _pf(pf), _targetBurntRatio(targetBurntRatio){}
  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override;
};

} // sparsification
} // shm
} // kaminpar

#endif //FORSTFIRESAMPLER_H
