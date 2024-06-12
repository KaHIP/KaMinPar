#pragma once

#include <networkit/auxiliary/Multiprecision.hpp>
#include <networkit/edgescores/EdgeScore.hpp>

#include "ScoreBacedSampler.h"

namespace kaminpar::shm::sparsification {
template <typename Score> class NetworKitScoreAdapter : public ScoreFunction<Score> {
public:
  NetworKitScoreAdapter(std::function<NetworKit::EdgeScore<Score>(NetworKit::Graph)> constructor)
      : _curried_constructor(constructor) {}
  StaticArray<Score> scores(const CSRGraph &g) override;

private:
  std::function<NetworKit::EdgeScore<Score>(NetworKit::Graph)> _curried_constructor;
};
} // namespace kaminpar::shm::sparsification
