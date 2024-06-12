#pragma once
#include "Sampler.h"
#include "ScoreBacedSampler.h"

namespace kaminpar::shm::sparsification {

template <typename Score> class ThresholdSampler : public ScoreBacedSampler<Score> {
public:
  ThresholdSampler(
      std::unique_ptr<ScoreFunction<Score>> scoreFunction,
      std::unique_ptr<ReweighingFunction<Score>> reweighingFunction
  )
      : ScoreBacedSampler<Score>(std::move(scoreFunction), std::move(reweighingFunction)) {}
  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override {
    auto sample = StaticArray<EdgeWeight>(g.m(), 0);
    auto scores = this->_score_function->scores(g);
    Score threshold = find_theshold(scores, target_edge_amount);
    tbb::parallel_for(static_cast<EdgeID>(0), g.m() - 1, [&](EdgeID e) {
      sample[e] = scores[e] >= threshold
                      ? this->_reweighing_function->new_weight(g.edge_weight(e), scores[e])
                      : 0;
    });
    return sample;
  }

private:
  EdgeID find_theshold(const StaticArray<Score> &scores, EdgeID target_edge_amount) {
    StaticArray<Score> sorted_scores(scores.begin(), scores.end());
    std::sort(sorted_scores.begin(), sorted_scores.end());
    return sorted_scores[target_edge_amount];
  };
};
} // namespace kaminpar::shm::sparsification
