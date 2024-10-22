#pragma once
#include <ranges>

#include "ScoreBacedSampler.h"
#include "sparsification_utils.h"

#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm::sparsification {
template <typename Score> class IndependentRandomSampler : public ScoreBacedSampler<Score> {
public:
  IndependentRandomSampler(std::unique_ptr<ScoreFunction<Score>> scoreFunction)
      : ScoreBacedSampler<Score>(std::move(scoreFunction)) {}

  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override {
    auto scores = this->_score_function->scores(g);
    double factor = normalizationFactor(g, scores, target_edge_amount);

    StaticArray<EdgeWeight> sample(g.m(), 0);
    utils::for_upward_edges(g, [&](EdgeID e) {
      sample[e] = Random::instance().random_bool(factor * scores[e]) ? g.edge_weight(e) : 0;
    });
    return sample;
  }

  double normalizationFactor(const CSRGraph &g, const StaticArray<Score> &scores, EdgeID target) {
    StaticArray<Score> sorted_scores(g.m() / 2);
    StaticArray<Score> prefix_sum(g.m() / 2);
    EdgeID end_of_sorted_scores = 0;
    utils::for_upward_edges(g, [&](EdgeID e) {
      sorted_scores[end_of_sorted_scores++] = static_cast<Score>(scores[e]);
    });
    std::sort(sorted_scores.begin(), sorted_scores.end());
    parallel::prefix_sum(sorted_scores.begin(), sorted_scores.end(), prefix_sum.begin());

    auto expected_at_index = [&](EdgeID i) {
      return g.m() / 2 - i - 1 + 1 / static_cast<double>(sorted_scores[i]) * prefix_sum[i];
    };

    auto possible_indices =
        std::ranges::iota_view(static_cast<EdgeID>(0), g.m() / 2) | std::views::reverse;
    EdgeID index = *std::upper_bound(
        possible_indices.begin(),
        possible_indices.end(),
        target / 2,
        [&](EdgeID t, NodeID i) {
          return t <= expected_at_index(i); // negated to make asc
        }
    );

    double factor = static_cast<double>((target / 2 - (g.m() / 2 - index))) / prefix_sum[index - 1];


    return factor;
  }
};
}; // namespace kaminpar::shm::sparsification