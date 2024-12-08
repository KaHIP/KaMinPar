#pragma once
#include <ranges>

#include "ScoreBacedSampler.h"
#include "sparsification_utils.h"

#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm::sparsification {
template <typename Score> class IndependentRandomSampler : public ScoreBacedSampler<Score> {
public:
  IndependentRandomSampler(
      std::unique_ptr<ScoreFunction<Score>> scoreFunction, bool noApprox = false
  )
      : ScoreBacedSampler<Score>(std::move(scoreFunction)),
        _noApprox(noApprox) {}

  double normalizationFactor(const CSRGraph &g, const StaticArray<Score> &scores, EdgeID target);
  double
  exactNormalizationFactor(const CSRGraph &g, const StaticArray<Score> &scores, EdgeID target);
  double
  approxNormalizationFactor(const CSRGraph &g, const StaticArray<Score> &scores, EdgeID target);

  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override;

  static EdgeID exponential_bucket(EdgeWeight score) {
    return 31 - __builtin_clz(score);
  }

private:
  bool _noApprox;
};

template class IndependentRandomSampler<EdgeWeight>;
template class IndependentRandomSampler<EdgeID>;
template class IndependentRandomSampler<double>;
}; // namespace kaminpar::shm::sparsification