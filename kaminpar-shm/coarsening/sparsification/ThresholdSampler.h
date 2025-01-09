#pragma once
#include <oneapi/tbb/parallel_sort.h>
#include <utils/output.hpp>

#include "Sampler.h"
#include "ScoreBacedSampler.h"
#include "ips4o.hpp"
#include "sparsification_utils.h"

#include "kaminpar-common/timer.h"

namespace kaminpar::shm::sparsification {

template <typename Score> class ThresholdSampler : public ScoreBacedSampler<Score> {
public:
  ThresholdSampler(std::unique_ptr<ScoreFunction<Score>> scoreFunction)
      : ScoreBacedSampler<Score>(std::move(scoreFunction)) {}

  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override {
    SCOPED_TIMER("Threshold Sampling");
    auto sample = StaticArray<EdgeWeight>(g.m(), 0);
    StaticArray<Score> scores;
    {
      SCOPED_TIMER("Calculate Scores");
      scores = this->_score_function->scores(g);
    }

      utils::K_SmallestInfo<Score> threshold;
      {
        SCOPED_TIMER("Find Threshold with iso4o sort");
        threshold =
          find_threshold(scores,target_edge_amount);
      }

      double inclusion_probaility_if_equal = (target_edge_amount / 2 - threshold.number_of_elements_smaller) / static_cast<double>(threshold.number_of_elemtns_equal);
      utils::parallel_for_upward_edges(g, [&](EdgeID e) {
        if (scores[e] < threshold.value || (scores[e] == threshold.value && Random::instance().random_bool(inclusion_probaility_if_equal))) {
          sample[e] = g.edge_weight(e);
        }
      });

    return sample;
  }

private:
  utils::K_SmallestInfo<Score>
  find_threshold(const StaticArray<Score> &scores, EdgeID target_edge_amount) {
    utils::K_SmallestInfo<Score> output;
    StaticArray<Score> sorted_scores(scores.begin(), scores.end());
    ips4o::parallel::sort(sorted_scores.begin(), sorted_scores.end());

    EdgeID indexOfThreshold = sorted_scores.size() - target_edge_amount;
    output.value = sorted_scores[indexOfThreshold];
    EdgeID indexOfFirstLargerScore =
        std::upper_bound(sorted_scores.begin(), sorted_scores.end(), output.value) -
        sorted_scores.begin();
    EdgeID indexOfFirstEqualScore =
        std::lower_bound(sorted_scores.begin(), sorted_scores.end(), output.value) -
        sorted_scores.begin();
    EdgeID numEdgesAtThresholdScoreToInclude = (indexOfFirstLargerScore - indexOfThreshold) / 2;
    output.number_of_elemtns_equal = (indexOfFirstLargerScore - indexOfFirstEqualScore)/2;
    output.number_of_elements_smaller = indexOfFirstEqualScore/2;
    return output;
  };
};
} // namespace kaminpar::shm::sparsification
