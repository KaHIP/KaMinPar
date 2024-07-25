#pragma once
#include "IndexDistributionWithReplacement.h"
#include "ScoreBacedSampler.h"
#include "sparsification_utils.h"

namespace kaminpar::shm::sparsification {
template <typename Score> class RandomWithReplacementSampler : public ScoreBacedSampler<Score> {
public:
  RandomWithReplacementSampler(
      std::unique_ptr<ScoreFunction<Score>> scoreFunction,
      std::unique_ptr<ReweighingFunction<Score>> reweighingFunction
  )
      : ScoreBacedSampler<Score>(std::move(scoreFunction), std::move(reweighingFunction)) {}

  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override {
    auto scores = this->_score_function->scores(g);
    utils::for_downward_edges(g, [&](EdgeID e) { scores[e] = 0; });
    auto distribution = IndexDistributionWithReplacement<Score>(scores.begin(), scores.end());

    EdgeID edges_sampled = 0;
    EdgeID iterations = 0;
    StaticArray<EdgeWeight> sample(g.m(), 0);
    while (iterations  < target_edge_amount) {
      EdgeID e = distribution();
      if (sample[e] == 0) // new edge
        edges_sampled++;

      sample[e] += this->_reweighing_function->new_weight(g.edge_weight(e), scores[e]);
      iterations++;
    }

    return sample;
  }
};
} // namespace kaminpar::shm::sparsification