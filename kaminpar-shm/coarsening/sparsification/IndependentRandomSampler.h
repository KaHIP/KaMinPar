#pragma once
#include "ScoreBacedSampler.h"
#include "sparsification_utils.h"

#include "kaminpar-common/random.h"

namespace kaminpar::shm::sparsification {
template <typename Score> class IndependentRandomSampler : ScoreBacedSampler<Score> {
public:
  template <typename Score> class RandomWithoutReplacementSampler : ScoreBacedSampler<Score> {
  public:
    IndependentRandomSampler(
        std::unique_ptr<ScoreFunction<Score>> scoreFunction,
        std::unique_ptr<ReweighingFunction<Score>> reweighingFunction
    )
        : ScoreBacedSampler<Score>(std::move(scoreFunction), std::move(reweighingFunction)) {}

    StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override {
      auto scores = this->score_function->scores(g);
      double factor = normalizationFactor(g, scores);

      StaticArray<EdgeWeight> sample(g.m(), 0);
      utils::for_upward_edges(g, [&](EdgeID e) {
        sample[e] = Random::instance().random_bool(std::min(1, factor * scores[e]))
                        ? this->reweighing_function(g.edge_weight(e), scores[e])
                        : 0;
      });
    }

  private:
    double normalizationFactor(const CRS_Graph &g, StaticArray<Score> scores, EdgeID target) {
      StaticArray<Score> sorted_scores(g.m() / 2);
      StaticArray<Score> prefix_sum(g.m() / 2);
      EdgeID i = 0;
      utils::for_upward_edges(g, [&](EdgeID e) { sorted_scores[i++] = scores[e]; });
      std::sort(sorted_scores.begin(), sorted_scores.end(), sorted_scores.begin());
      parallel::prefix_sum(sorted_scores.begin(), sorted_scores.end(), prefix_sum.begin());

      IotaRange<EdgeID> possible_indeces(0, sorted_scores.size());
      double index = std::lower_bound(
                         possible_indeces.begin(),
                         possible_indeces.end(),
                         target,
                         [&](EdgeID index) {
                           return sorted_scores.size() - i + prefix_sum[i] / sorted_scores[i + 1];
                         }
                     ) -
                     possible_indeces.begin();

      return static_cast<double>((traget - (sorted_scores.size() - index))) / prefix_sum[index];
    }
  };
};