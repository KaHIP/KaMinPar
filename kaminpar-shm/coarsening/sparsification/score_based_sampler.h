#pragma once

#include "kaminpar-shm/coarsening/sparsification/sampler.h"
#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::sparsification {

template <typename Score> class ScoreFunction {
public:
  virtual ~ScoreFunction() = default;

  virtual StaticArray<Score> scores(const CSRGraph &g) = 0;
};

template <typename Score> class ReweighingFunction {
public:
  virtual ~ReweighingFunction() = default;

  virtual EdgeWeight new_weight(EdgeWeight old_weight, Score score) = 0;
};

template <typename Score> class WeightDiviedByScore : public ReweighingFunction<Score> {
  EdgeWeight new_weight(const EdgeWeight old_weight, const Score score) override {
    return old_weight / score;
  }
};

template <typename Score> class ScoreBasedSampler : public Sampler {
public:
  ScoreBasedSampler(std::unique_ptr<ScoreFunction<Score>> score_function)
      : _score_function(std::move(score_function)) {}

  virtual ~ScoreBasedSampler() = default;

protected:
  std::unique_ptr<ScoreFunction<Score>> _score_function;
};

} // namespace kaminpar::shm::sparsification
