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
    utils::parallel_for_upward_edges(g, [&](EdgeID e) {
      sample[e] = Random::instance().random_bool(factor * scores[e]) ? g.edge_weight(e) : 0;
    });
    return sample;
  }

private:
  static EdgeID exponential_bucket(Score score) {
    return 31 - __builtin_clz(score);
  }
  double normalizationFactor(const CSRGraph &g, const StaticArray<Score> &scores, EdgeID target) {
    EdgeID number_of_buckets = exponential_bucket(g.total_edge_weight()) + 1;
    std::vector<tbb::concurrent_vector<Score>> expontial_buckets(number_of_buckets);
    StaticArray<Score> buckets_score_prefixsum(number_of_buckets);
    StaticArray<EdgeID> buckets_size_prefixsum(number_of_buckets);
    tbb::parallel_for(static_cast<EdgeID>(0), g.m(), [&](EdgeID e) {
      Score score = g.edge_weight(e);
      auto bucket = exponential_bucket(score);
      expontial_buckets[bucket].push_back(e);
      __atomic_add_fetch(&buckets_score_prefixsum[bucket], score, __ATOMIC_RELAXED);
    });
    parallel::prefix_sum(
        buckets_score_prefixsum.begin(),
        buckets_score_prefixsum.end(),
        buckets_score_prefixsum.begin()
    );
    for (EdgeID i = 0; i < number_of_buckets; i++) {
      buckets_size_prefixsum[i] = expontial_buckets[i].size();
    }
    parallel::prefix_sum(
        buckets_size_prefixsum.begin(), buckets_size_prefixsum.end(), buckets_size_prefixsum.begin()
    );

    auto possible_buckets = std::ranges::iota_view(static_cast<EdgeID>(0), number_of_buckets);
    auto bucket_index = *std::upper_bound(
        possible_buckets.begin(),
        possible_buckets.end(),
        target,
        [&](EdgeID target, auto bucket_index) {
          return target <= g.m() - buckets_size_prefixsum[bucket_index] +
                               ((1 << bucket_index) - 1) * buckets_score_prefixsum[bucket_index];
        }
    );

    double new_normalization_factor = (target - (g.m() - buckets_size_prefixsum[bucket_index])) /
                                      static_cast<double>(buckets_score_prefixsum[bucket_index]);

    // Old algo
    StaticArray<Score> sorted_scores(g.m() / 2);
    StaticArray<Score> prefix_sum(g.m() / 2);
    EdgeID i = 0;
    utils::for_upward_edges(g, [&](EdgeID e) { sorted_scores[i++] = scores[e]; });
    tbb::parallel_sort(sorted_scores.begin(), sorted_scores.end());
    parallel::prefix_sum(sorted_scores.begin(), sorted_scores.end(), prefix_sum.begin());

    EdgeID upper = 0;
    EdgeID lower = sorted_scores.size();
    while (lower + 1 < upper) {
      EdgeID mid = lower + (upper - lower) / 2;
      if (target < (sorted_scores.size() - mid + prefix_sum[mid] / sorted_scores[mid + 1]))
        upper = mid;
      else
        lower = mid;
    }
    EdgeID index = lower;

    double old_normalization =
        static_cast<double>((target - (sorted_scores.size() - index))) / prefix_sum[index];
    printf(
        "*** Normalization: old = %f, new %f ***\n", old_normalization, new_normalization_factor
    );
    return new_normalization_factor;
  }
};
}; // namespace kaminpar::shm::sparsification