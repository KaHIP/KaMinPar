#pragma once
#include "ScoreBacedSampler.h"

namespace kaminpar::shm::sparsification {
class WeightedForestFireScore : public ScoreFunction<EdgeID> {
public:
  WeightedForestFireScore(double pf, double targetBurnRatio)
      : _pf(pf),
        _targetBurnRatio(targetBurnRatio) {}
  StaticArray<EdgeID> scores(const CSRGraph &g) override;

private:
  double _pf;
  double _targetBurnRatio;
};
} // namespace kaminpar::shm::sparsification
