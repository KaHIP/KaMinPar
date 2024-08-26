#pragma once
#include "IndexDistributionWithReplacement.h"
#include "ScoreBacedSampler.h"
#include "sparsification_utils.h"

namespace kaminpar::shm::sparsification {
template <typename Score> class RandomWithReplacementSampler : public ScoreBacedSampler<Score> {
public:
  RandomWithReplacementSampler(
      std::unique_ptr<ScoreFunction<Score>> scoreFunction
  )
      : ScoreBacedSampler<Score>(std::move(scoreFunction)) {}

  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override {
    auto scores = this->_score_function->scores(g);
    utils::for_downward_edges(g, [&](EdgeID e) { scores[e] = 0; });

    EdgeWeight totalWeight = 0;
    utils::for_upward_edges(g, [&](EdgeID e) { totalWeight += g.edge_weight(e); });
    auto distribution = IndexDistributionWithReplacement(scores.begin(), scores.end());

    EdgeID edges_sampled = 0;
    EdgeID iterations = 0;
    StaticArray<EdgeWeight> sample(g.m(), 0);
    while (iterations < target_edge_amount) {
      EdgeID e = distribution();
      if (sample[e] == 0) // new edge
        edges_sampled++;

      sample[e] += totalWeight / target_edge_amount;
      iterations++;
    }

    return sample;
  }
};
} // namespace kaminpar::shm::sparsification