#pragma once

#include "kaminpar-shm/coarsening/sparsification/index_distribution_with_replacement.h"
#include "kaminpar-shm/coarsening/sparsification/score_based_sampler.h"
#include "kaminpar-shm/coarsening/sparsification/sparsification_utils.h"

namespace kaminpar::shm::sparsification {

template <typename Score> class RandomWithReplacementSampler : public ScoreBasedSampler<Score> {
public:
  RandomWithReplacementSampler(std::unique_ptr<ScoreFunction<Score>> score_function)
      : ScoreBasedSampler<Score>(std::move(score_function)) {}

  StaticArray<EdgeWeight> sample(const CSRGraph &g, const EdgeID target_edge_amount) override {
    auto scores = this->_score_function->scores(g);
    utils::for_downward_edges(g, [&](const EdgeID e) { scores[e] = 0; });

    EdgeWeight totalWeight = 0;
    utils::for_upward_edges(g, [&](const EdgeID e) { totalWeight += g.edge_weight(e); });
    auto distribution = IndexDistributionWithReplacement(scores.begin(), scores.end());

    EdgeID iterations = 0;
    StaticArray<EdgeWeight> sample(g.m(), 0);

    while (iterations < target_edge_amount) {
      const EdgeID e = distribution();
      sample[e] += totalWeight / target_edge_amount;
      iterations++;
    }

    return sample;
  }
};

} // namespace kaminpar::shm::sparsification
