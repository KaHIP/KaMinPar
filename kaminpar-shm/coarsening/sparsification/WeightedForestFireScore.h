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

private:
  double _pf;
  double _targetBurnRatio;

  void print_fire_statistics(
      const CSRGraph &g,
      EdgeID edges_burnt,
      int number_of_fires,
      tbb::concurrent_vector<EdgeID> numbers_of_edges_burnt
  );
};
} // namespace kaminpar::shm::sparsification
