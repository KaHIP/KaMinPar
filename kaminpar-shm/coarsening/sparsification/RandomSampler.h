#pragma once
#include "FiniteRandomDistribution.h"
#include "ScoreBacedSampler.h"
#include "sparsification_utils.h"

namespace kaminpar::shm::sparsification {
template <typename Score> class RandomSampler : public ScoreBacedSampler<Score> {

  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override {
    auto scores = _score_function->scores(g);
    utils::for_downward_edges(g, [&](EdgeID e) { scores[e] = 0; });
    auto distribution = FiniteRandomDistribution<Score>(scores);

    EdgeID edges_sampled = 0;
    StaticArray<EdgeWeight> sample(g.m(), 0);
    while (edges_sampled < target_edge_amount) {
      EdgeID e = distribution();
      if (sample[e] == 0) // new edge
        edges_sampled++;

      sample[e] += this->_reweighing_function->new_weight(g.edge_weight(e), scores[e]);
    }

    return sample;
  }
};
} // namespace kaminpar::shm::sparsification