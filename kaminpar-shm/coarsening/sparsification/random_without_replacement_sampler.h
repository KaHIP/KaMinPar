#pragma once

#include "kaminpar-shm/coarsening/sparsification/index_distribution_without_replacement.h"
#include "kaminpar-shm/coarsening/sparsification/score_based_sampler.h"
#include "kaminpar-shm/coarsening/sparsification/sparsification_utils.h"

namespace kaminpar::shm::sparsification {

template <typename Score> class RandomWithoutReplacementSampler : public ScoreBasedSampler<Score> {
public:
  RandomWithoutReplacementSampler(std::unique_ptr<ScoreFunction<Score>> score_function)
      : ScoreBasedSampler<Score>(std::move(score_function)) {}

  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override {
    auto scores = this->_score_function->scores(g);
    utils::for_downward_edges(g, [&](EdgeID e) { scores[e] = 0; });
    IndexDistributionWithoutReplacement distribution(scores.begin(), scores.end());

    StaticArray<EdgeWeight> sample(g.m(), 0);
    for (EdgeID edges_sampled = 0; edges_sampled < target_edge_amount / 2; edges_sampled++) {
      EdgeID e = distribution();
      KASSERT(sample[e] == 0, "Sampling WITH and not WITHOUT replacement", assert::always);
      sample[e] = g.edge_weight(e);
    }
    return sample;
  }
};

} // namespace kaminpar::shm::sparsification
