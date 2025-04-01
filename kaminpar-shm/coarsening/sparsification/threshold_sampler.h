#pragma once

#include "kaminpar-shm/coarsening/sparsification/score_based_sampler.h"
#include "kaminpar-shm/coarsening/sparsification/sparsification_utils.h"

#include "kaminpar-common/random.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::sparsification {

template <typename Score> class ThresholdSampler : public ScoreBasedSampler<Score> {
public:
  ThresholdSampler(std::unique_ptr<ScoreFunction<Score>> score_function)
      : ScoreBasedSampler<Score>(std::move(score_function)) {}

  virtual StaticArray<EdgeWeight>
  sample(const CSRGraph &g, const EdgeID target_edge_amount) override {
    SCOPED_TIMER("Threshold Sampling");

    StaticArray<EdgeWeight> sample(g.m());
    StaticArray<Score> scores = TIMED_SCOPE("Calculate scores") {
      return this->_score_function->scores(g);
    };

    auto threshold = TIMED_SCOPE("Find threshold with qselect") {
      return utils::quickselect_k_smallest<Score>(
          g.m() - target_edge_amount + 1, scores.begin(), scores.end()
      );
    };

    const EdgeID number_of_elements_larger =
        g.m() - threshold.number_of_elements_equal - threshold.number_of_elements_smaller;
    KASSERT(number_of_elements_larger <= target_edge_amount, "quickselect failed", assert::always);
    const EdgeID number_of_equal_elements_to_include =
        target_edge_amount - number_of_elements_larger;
    const double inclusion_probability_if_equal =
        number_of_equal_elements_to_include /
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
