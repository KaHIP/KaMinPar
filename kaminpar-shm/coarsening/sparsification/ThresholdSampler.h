#pragma once

#include "kaminpar-shm/coarsening/sparsification/ScoreBacedSampler.h"
#include "kaminpar-shm/coarsening/sparsification/sparsification_utils.h"

#include "kaminpar-common/timer.h"

namespace kaminpar::shm::sparsification {

template <typename Score> class ThresholdSampler : public ScoreBacedSampler<Score> {
public:
  ThresholdSampler(std::unique_ptr<ScoreFunction<Score>> scoreFunction)
      : ScoreBacedSampler<Score>(std::move(scoreFunction)) {}

  void sample2(CSRGraph &graph, EdgeID target_edge_amount) override {
    SCOPED_TIMER("Threshold Sampling");

    const utils::K_SmallestInfo<EdgeWeight> threshold = TIMED_SCOPE("Threshold selection") {
      return utils::quickselect_k_smallest<EdgeWeight>(
          target_edge_amount, graph.raw_edge_weights().begin(), graph.raw_edge_weights().end()
      );
    };

    TIMED_SCOPE("Edge selection") {
      const double inclusion_probability_if_equal =
          (target_edge_amount - threshold.number_of_elements_smaller) /
          static_cast<double>(threshold.number_of_elemtns_equal);

      utils::parallel_for_upward_edges(graph, [&](const EdgeID e) {
        if (graph.edge_weight(e) < threshold.value ||
            (graph.edge_weight(e) == threshold.value &&
             Random::instance().random_bool(inclusion_probability_if_equal))) {
          graph.raw_edge_weights()[e] *= -1;
        }
      });
    };
  }

  StaticArray<EdgeWeight> sample(const CSRGraph &, EdgeID) override {
    return {};
  }
};

} // namespace kaminpar::shm::sparsification
