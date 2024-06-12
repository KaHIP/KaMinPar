#pragma once
#include "Sampler.h"

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
namespace kaminpar::shm::sparsification {
template <typename Score> class ScoreFunction {
public:
  virtual StaticArray<Score> scores(const CSRGraph &g) = 0;
};
template <typename Score> class ReweighingFunction {
public:
  virtual EdgeWeight new_weight(EdgeWeight old_weight, Score score) = 0;
};
template <typename Score> class IdentityReweihingFunction : public ReweighingFunction<Score> {
  EdgeWeight new_weight(EdgeWeight old_weight, Score score) override {
    return old_weight;
  }
};
template <typename Score> class ScoreBacedSampler : public Sampler {
public:
  ScoreBacedSampler(
      std::unique_ptr<ScoreFunction<Score>> scoreFunction,
      std::unique_ptr<ReweighingFunction<Score>> reweighing_function
  )
      : _score_function(std::move(scoreFunction)),
        _reweighing_function(std::move(reweighing_function)){};

protected:
  std::unique_ptr<ScoreFunction<Score>> _score_function;
  std::unique_ptr<ReweighingFunction<Score>> _reweighing_function;
};
} // namespace kaminpar::shm::sparsification
