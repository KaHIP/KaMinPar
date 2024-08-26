#pragma once
#include "IndexDistributionWithoutReplacement.h"
#include "ScoreBacedSampler.h"
#include "sparsification_utils.h"

namespace kaminpar::shm::sparsification {
template <typename Score> class RandomWithoutReplacementSampler : public ScoreBacedSampler<Score> {
public:
  RandomWithoutReplacementSampler(
      std::unique_ptr<ScoreFunction<Score>> scoreFunction
  )
      : ScoreBacedSampler<Score>(std::move(scoreFunction)) {}

  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override {
    auto scores = this->_score_function->scores(g);
    utils::for_downward_edges(g, [&](EdgeID e) { scores[e] = 0; });
    IndexDistributionWithoutReplacement distribution(scores.begin(), scores.end());

    StaticArray<EdgeWeight> sample(g.m(), 0);
    for (int edges_sampled = 0; edges_sampled < target_edge_amount / 2; edges_sampled++) {
      EdgeID e = distribution();
      KASSERT(sample[e] == 0, "Sampling WITH and not WITHOUT replacement", assert::always);
      sample[e] = g.edge_weight(e);
    }
    return sample;
  }
};
} // namespace kaminpar::shm::sparsification