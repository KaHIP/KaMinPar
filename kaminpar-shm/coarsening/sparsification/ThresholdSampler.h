#pragma once
#include "Sampler.h"
#include "ScoreBacedSampler.h"

namespace kaminpar::shm::sparsification {
template <typename T> class ThresholdSampler : public ScoreBacedSampler<T> {
public:
  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override;
private:
  EdgeID find_theshold(const StaticArray<T> scores, EdgeID target_edge_amount);
};
} // namespace kaminpar::shm::sparsification
