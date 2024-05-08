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
  float _threshold;
public:
  ForestFireSampler(float pf, float targetBurntRatio, float threshold): _pf(pf), _targetBurntRatio(targetBurntRatio), _threshold(threshold){}
  StaticArray<EdgeWeight> sample(const CSRGraph &g) override;
};

} // sparsification
} // shm
} // kaminpar

#endif //FORSTFIRESAMPLER_H
