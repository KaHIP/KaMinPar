#pragma once
#include "Sampler.h"

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
namespace kaminpar::shm::sparsification {
template <typename Score> class ScoreFunction {
public:
  virtual StaticArray<Score> scores(const CSRGraph &g) = 0;
  virtual ~ScoreFunction() = default;
};
template <typename Score> class ReweighingFunction {
public:
  virtual EdgeWeight new_weight(EdgeWeight old_weight, Score score) = 0;
  virtual ~ReweighingFunction() = default;
};
template <typename Score> class WeightDiviedByScore : public ReweighingFunction<Score> {
  EdgeWeight new_weight(EdgeWeight old_weight, Score score) override {
    return old_weight / score;
  }
};
template <typename Score> class ScoreBacedSampler : public Sampler {
public:
  ScoreBacedSampler(
      std::unique_ptr<ScoreFunction<Score>> scoreFunction
  )
      : _score_function(std::move(scoreFunction)){};
  virtual ~ScoreBacedSampler() = default;

protected:
  std::unique_ptr<ScoreFunction<Score>> _score_function;
};
} // namespace kaminpar::shm::sparsification
