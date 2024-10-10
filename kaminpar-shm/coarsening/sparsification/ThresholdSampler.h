#pragma once
#include <oneapi/tbb/parallel_sort.h>

#include "Sampler.h"
#include "ScoreBacedSampler.h"
#include "sparsification_utils.h"

#include "kaminpar-common/timer.h"

namespace kaminpar::shm::sparsification {

template <typename Score> class ThresholdSampler : public ScoreBacedSampler<Score> {
public:
  ThresholdSampler(std::unique_ptr<ScoreFunction<Score>> scoreFunction, bool noApprox = false)
      : ScoreBacedSampler<Score>(std::move(scoreFunction)),
        _noApprox(noApprox) {}

  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override {
    SCOPED_TIMER("Threshold Sampling");
    auto sample = StaticArray<EdgeWeight>(g.m(), 0);
    StaticArray<Score> scores;
    {
      SCOPED_TIMER("Calculate Scores");
      scores = this->_score_function->scores(g);
    }

    if (_noApprox) {
      auto [threshold, numEdgesAtThresholdScoreToInclude] =
          find_threshold(scores, target_edge_amount);

      utils::parallel_for_upward_edges(g, [&](EdgeID e) {
        if (scores[e] > threshold) {
          sample[e] = g.edge_weight(e);
        } else if (scores[e] == threshold && numEdgesAtThresholdScoreToInclude > 0) {
          sample[e] = g.edge_weight(e);
          __atomic_add_fetch(&numEdgesAtThresholdScoreToInclude, -1, __ATOMIC_RELAXED);
        }
      });

      KASSERT(
          numEdgesAtThresholdScoreToInclude == 0,
          "not all nessary edges with threshold score included"
      );
    } else {
      Score threshold;
      {
        SCOPED_TIMER("Find Threshold with qselect");
        threshold =
            utils::quickselect_k_smallest<Score>(target_edge_amount, scores.begin(), scores.end());
      }
      utils::parallel_for_upward_edges(g, [&](EdgeID e) {
        if (scores[e] <= threshold)
          sample[e] = g.edge_weight(e);
      });
    }
    return sample;
  }

private:
  std::pair<EdgeID, EdgeID>
  find_threshold(const StaticArray<Score> &scores, EdgeID target_edge_amount) {
    SCOPED_TIMER("Find Threshold");
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
  bool _noApprox;
};
} // namespace kaminpar::shm::sparsification
