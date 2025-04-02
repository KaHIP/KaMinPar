#pragma once
#include <oneapi/tbb/concurrent_vector.h>

#include "ScoreBacedSampler.h"

namespace kaminpar::shm::sparsification {
class WeightedForestFireScore : public ScoreFunction<EdgeID> {
public:
  WeightedForestFireScore(double pf, double targetBurnRatio)
      : _pf(pf),
        _targetBurnRatio(targetBurnRatio) {}
  StaticArray<EdgeID> scores(const CSRGraph &g) override;

  static void make_scores_symetric(const CSRGraph &g, StaticArray<EdgeID> &scores);

private:
  double _pf;
  double _targetBurnRatio;

  void print_fire_statistics(const CSRGraph &g, EdgeID edges_burnt, int number_of_fires);
};
} // namespace kaminpar::shm::sparsification
