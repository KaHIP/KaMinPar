#pragma once
#include "Sampler.h"
#include "ScoreBacedSampler.h"
#include "sparsification_utils.h"
    #include <oneapi/tbb/parallel_sort.h>

namespace kaminpar::shm::sparsification {

template <typename Score> class ThresholdSampler : public ScoreBacedSampler<Score> {
public:
  ThresholdSampler(
      std::unique_ptr<ScoreFunction<Score>> scoreFunction
  )
      : ScoreBacedSampler<Score>(std::move(scoreFunction)) {}

  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override {
    auto sample = StaticArray<EdgeWeight>(g.m(), 0);
    auto scores = this->_score_function->scores(g);
    auto [threshold, numEdgesAtThresholdScoreToInclude] =
        find_threshold(scores, target_edge_amount, g);

    utils::p_for_upward_edges(g, [&](EdgeID e) {
      if (scores[e] > threshold) {
        sample[e] = g.edge_weight(e);
      } else if (scores[e] == threshold && numEdgesAtThresholdScoreToInclude > 0) {
        sample[e] = g.edge_weight(e);
        numEdgesAtThresholdScoreToInclude--;
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
    std::vector<Score> sorted_scores(scores.size());
    tbb::parallel_for(0ul, scores.size(), [&](auto e) { sorted_scores[e] = scores[e]; });
    tbb::parallel_sort(sorted_scores.begin(), sorted_scores.end());

    EdgeID indexOfThreshold = sorted_scores.size() - target_edge_amount;
    Score threshold = sorted_scores[indexOfThreshold];
    EdgeID indexOfFirstLagerScore =
        std::upper_bound(sorted_scores.begin(), sorted_scores.end(), threshold) -
        sorted_scores.begin();
    EdgeID numEdgesAtThresholdScoreToInclude = indexOfFirstLagerScore - indexOfThreshold / 2;
    return std::make_pair(threshold, numEdgesAtThresholdScoreToInclude);
  };
};
} // namespace kaminpar::shm::sparsification
