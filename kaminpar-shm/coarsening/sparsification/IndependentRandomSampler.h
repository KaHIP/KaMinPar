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

  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override {
    auto scores = this->_score_function->scores(g);
    double factor = exactNormalizationFactor(g, scores, target_edge_amount);

    StaticArray<EdgeWeight> sample(g.m(), 0);
    utils::parallel_for_upward_edges(g, [&](EdgeID e) {
      sample[e] = Random::instance().random_bool(factor * scores[e]) ? g.edge_weight(e) : 0;
    });
    return sample;
  }

  static EdgeID exponential_bucket(EdgeWeight score) {
    return 31 - __builtin_clz(score);
  }

  double
  normalizationFactor(const CSRGraph &g, const StaticArray<Score> &scores, EdgeID target);
  double
  exactNormalizationFactor(const CSRGraph &g, const StaticArray<Score> &scores, EdgeID target);
  double
  approxNormalizationFactor(const CSRGraph &g, const StaticArray<Score> &scores, EdgeID target);

private:
  bool _noApprox;
};
}; // namespace kaminpar::shm::sparsification