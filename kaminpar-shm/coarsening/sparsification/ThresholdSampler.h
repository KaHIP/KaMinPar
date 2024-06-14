#pragma once
#include "Sampler.h"
#include "ScoreBacedSampler.h"
#include "sparsification_utils.h"

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
    auto [threshold, numEdgesAtThresholdScoreToInclude] =
        find_threshold(scores, target_edge_amount, g);

    utils::for_upward_edges(g, [&](EdgeID e) {
      if (scores[e] > threshold) {
        sample[e] = this->_reweighing_function->new_weight(g.edge_weight(e), scores[e]);
      } else if (scores[e] == threshold && numEdgesAtThresholdScoreToInclude > 0) {
        sample[e] = this->_reweighing_function->new_weight(g.edge_weight(e), scores[e]);
        numEdgesAtThresholdScoreToInclude--;
      } else {
        sample[e] = 0;
      }
    });

    KASSERT(
        numEdgesAtThresholdScoreToInclude == 0,
        "not all nessary edges with threshold score included"
    );
    return sample;
  }

private:
  std::pair<EdgeID, EdgeID>
  find_threshold(const StaticArray<Score> &scores, EdgeID target_edge_amount, const CSRGraph &g) {
    StaticArray<Score> sorted_scores(scores.size() / 2);

    EdgeID i = 0;
    utils::for_upward_edges(g, [&](EdgeID e) { sorted_scores[i++] = scores[e]; });
    KASSERT(i == sorted_scores.size());

    std::sort(sorted_scores.begin(), sorted_scores.end());
    EdgeID indexOfThreshold = sorted_scores.size() - target_edge_amount / 2;
    Score threshold = sorted_scores[indexOfThreshold];
    EdgeID indexOfFirstLagerScore =
        std::upper_bound(sorted_scores.begin(), sorted_scores.end(), threshold) -
        sorted_scores.begin();
    EdgeID numEdgesAtThresholdScoreToInclude = indexOfFirstLagerScore - indexOfThreshold;
    return std::make_pair(threshold, numEdgesAtThresholdScoreToInclude);
  };
};
} // namespace kaminpar::shm::sparsification
