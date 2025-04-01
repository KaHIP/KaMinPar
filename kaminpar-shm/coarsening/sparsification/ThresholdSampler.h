#pragma once

#include "kaminpar-shm/coarsening/sparsification/ScoreBacedSampler.h"
#include "kaminpar-shm/coarsening/sparsification/sparsification_utils.h"

#include "kaminpar-common/timer.h"

namespace kaminpar::shm::sparsification {

template <typename Score> class ThresholdSampler : public ScoreBacedSampler<Score> {
public:
  ThresholdSampler(std::unique_ptr<ScoreFunction<Score>> scoreFunction)
      : ScoreBacedSampler<Score>(std::move(scoreFunction)) {}

  virtual StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override {
    SCOPED_TIMER("Threshold Sampling");
    auto sample = StaticArray<EdgeWeight>(g.m(), 0);
    StaticArray<Score> scores;
    {
      SCOPED_TIMER("Calculate Scores");
      scores = this->_score_function->scores(g);
    }

    utils::K_SmallestInfo<Score> threshold;
    {
      SCOPED_TIMER("Find Threshold with qselect");
      threshold = utils::quickselect_k_smallest<Score>(
          g.m() - target_edge_amount + 1, scores.begin(), scores.end()
      );
    }
    EdgeID number_of_elements_larger =
        g.m() - threshold.number_of_elements_equal - threshold.number_of_elements_smaller;
    KASSERT(number_of_elements_larger <= target_edge_amount, "quickselect failed", assert::always);
    EdgeID number_of_equal_elements_to_include = target_edge_amount - number_of_elements_larger;
    double inclusion_probability_if_equal = number_of_equal_elements_to_include /
                                            static_cast<double>(threshold.number_of_elements_equal);

    utils::parallel_for_upward_edges(g, [&](EdgeID e) {
      if (scores[e] > threshold.value ||
          (scores[e] == threshold.value &&
           Random::instance().random_bool(inclusion_probability_if_equal))) {
        sample[e] = g.edge_weight(e);
      }
    });

    return sample;
  }
};

} // namespace kaminpar::shm::sparsification
