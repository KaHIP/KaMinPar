
#include "ThresholdSampler.h"
namespace kaminpar::shm::sparsification {
template <typename T>
StaticArray<EdgeWeight> ThresholdSampler<T>::sample(const CSRGraph &g, EdgeID target_edge_amount) {
  auto sample = StaticArray<EdgeWeight>(g.m(), 0);
  auto scores = this->_score_function.run(g);
  T threshold = find_theshold(scores, target_edge_amount);
  tbb::parallel_for(0, g.m(), [&](EdgeID e) {
    sample[e] =
        scores[e] >= threshold ? this->_reweighing_function.new_weight(g.edge_weight(e), scores[e]) : 0;
  });
  return sample;
}

template <typename T>
EdgeID ThresholdSampler<T>::find_theshold(const StaticArray<T> scores, EdgeID target_edge_amount) {
  auto sorted_scores = scores;
  std::sort(sorted_scores.begin(), sorted_scores.end());
  return sorted_scores[target_edge_amount];
}
} // namespace kaminpar::shm::sparsification
