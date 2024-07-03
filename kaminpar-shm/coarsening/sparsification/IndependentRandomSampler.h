#pragma once
#include "ScoreBacedSampler.h"
#include "sparsification_utils.h"

#include "kaminpar-common/random.h"

namespace kaminpar::shm::sparsification {
template <typename Score> class IndependentRandomSampler : public ScoreBacedSampler<Score> {
public:
  IndependentRandomSampler(
      std::unique_ptr<ScoreFunction<Score>> scoreFunction,
      std::unique_ptr<ReweighingFunction<Score>> reweighingFunction
  )
      : ScoreBacedSampler<Score>(std::move(scoreFunction), std::move(reweighingFunction)) {}

  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override {
    auto scores = this->_score_function->scores(g);
    double factor = normalizationFactor(g, scores, target_edge_amount);

    StaticArray<EdgeWeight> sample(g.m(), 0);
    utils::for_upward_edges(g, [&](EdgeID e) {
      sample[e] = Random::instance().random_bool(factor * scores[e])
                      ? this->_reweighing_function->new_weight(g.edge_weight(e), scores[e])
                      : 0;
    });
    return sample;
  }

private:
  double normalizationFactor(const CSRGraph &g, const StaticArray<Score> &scores, EdgeID target) {
    StaticArray<Score> sorted_scores(g.m() / 2);
    StaticArray<Score> prefix_sum(g.m() / 2);
    EdgeID i = 0;
    utils::for_upward_edges(g, [&](EdgeID e) { sorted_scores[i++] = scores[e]; });
    std::sort(sorted_scores.begin(), sorted_scores.end());
    parallel::prefix_sum(sorted_scores.begin(), sorted_scores.end(), prefix_sum.begin());

    EdgeID upper = 0;
    EdgeID lower = sorted_scores.size();
    while (lower + 1 < upper) {
      EdgeID mid = lower + (upper - lower) / 2;
      if (target < (sorted_scores.size() - mid + prefix_sum[mid] / sorted_scores[mid + 1]))
        upper = mid;
      else
        lower = mid;
    }
    EdgeID index = lower;

    return static_cast<double>((target - (sorted_scores.size() - index))) / prefix_sum[index];
  }
};
}; // namespace kaminpar::shm::sparsification