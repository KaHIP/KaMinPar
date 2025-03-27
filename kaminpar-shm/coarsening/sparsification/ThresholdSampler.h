#pragma once
#include <oneapi/tbb/parallel_sort.h>

#include "Sampler.h"
#include "ScoreBacedSampler.h"
#include "sparsification_utils.h"

#include "kaminpar-common/timer.h"

namespace kaminpar::shm::sparsification {

template <typename Score> class ThresholdSampler : public ScoreBacedSampler<Score> {
public:
  ThresholdSampler(std::unique_ptr<ScoreFunction<Score>> scoreFunction)
      : ScoreBacedSampler<Score>(std::move(scoreFunction)) {}

  StaticArray<std::uint8_t> sample2(const CSRGraph &g, EdgeID target_edge_amount) override {
    SCOPED_TIMER("Threshold Sampling");

    // StaticArray<Score> scores = TIMED_SCOPE("Score calculation") {
    //   return this->_score_function->scores(g);
    // };

    const utils::K_SmallestInfo<EdgeWeight> threshold = TIMED_SCOPE("Threshold selection") {
      return utils::quickselect_k_smallest<EdgeWeight>(
          target_edge_amount, g.raw_edge_weights().begin(), g.raw_edge_weights().end()
      );
    };

    const double inclusion_probability_if_equal =
        (target_edge_amount - threshold.number_of_elements_smaller) /
        static_cast<double>(threshold.number_of_elemtns_equal);

    StaticArray<std::uint8_t> sample(g.m());
    utils::parallel_for_upward_edges(g, [&](const EdgeID e) {
      if (g.edge_weight(e) < threshold.value ||
          (g.edge_weight(e) == threshold.value &&
           Random::instance().random_bool(inclusion_probability_if_equal))) {
        sample[e] = 1;
      }
    });

    return sample;
  }

  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override {
    SCOPED_TIMER("Threshold Sampling");

    // StaticArray<Score> scores = TIMED_SCOPE("Score calculation") {
    //   return this->_score_function->scores(g);
    // };

    const utils::K_SmallestInfo<EdgeWeight> threshold = TIMED_SCOPE("Threshold selection") {
      return utils::quickselect_k_smallest<EdgeWeight>(
          target_edge_amount, g.raw_edge_weights().begin(), g.raw_edge_weights().end()
      );
    };

    const double inclusion_probability_if_equal =
        (target_edge_amount - threshold.number_of_elements_smaller) /
        static_cast<double>(threshold.number_of_elemtns_equal);

    StaticArray<EdgeWeight> sample(g.m());
    utils::parallel_for_upward_edges(g, [&](const EdgeID e) {
      if (g.edge_weight(e) < threshold.value ||
          (g.edge_weight(e) == threshold.value &&
           Random::instance().random_bool(inclusion_probability_if_equal))) {
        sample[e] = g.edge_weight(e);
      }
    });

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
};
} // namespace kaminpar::shm::sparsification
