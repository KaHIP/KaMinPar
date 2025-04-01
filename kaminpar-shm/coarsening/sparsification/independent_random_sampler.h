#pragma once

#include "kaminpar-shm/coarsening/sparsification/score_based_sampler.h"

namespace kaminpar::shm::sparsification {

template <typename Score> class IndependentRandomSampler : public ScoreBasedSampler<Score> {
public:
  IndependentRandomSampler(
      std::unique_ptr<ScoreFunction<Score>> score_function, const bool no_approx = false
  )
      : ScoreBasedSampler<Score>(std::move(score_function)),
        _no_approx(no_approx) {}

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
  bool _no_approx;
};

template class IndependentRandomSampler<EdgeWeight>;
template class IndependentRandomSampler<EdgeID>;
template class IndependentRandomSampler<double>;

} // namespace kaminpar::shm::sparsification
