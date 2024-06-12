#pragma once
#include "Sampler.h"

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
namespace kaminpar::shm::sparsification {
template <typename Score> class ScoreFunction {
public:
  virtual StaticArray<Score> scores(CSRGraph &g) = 0;
};
template <typename Score> class ReweighingFunction {
public:
  virtual EdgeWeight new_weight(EdgeWeight old_weight, Score score) = 0;
};
template <typename Score> class ScoreBacedSampler : public Sampler {
  ScoreBacedSampler(ScoreFunction<Score> scoreFunction) : _score_function(scoreFunction){};

protected:
  ScoreFunction<Score> _score_function;
  ReweighingFunction<Score> _reweighing_function;
};
} // namespace kaminpar::shm::sparsification
