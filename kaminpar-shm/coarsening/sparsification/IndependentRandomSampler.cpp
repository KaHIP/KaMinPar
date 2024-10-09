#include "IndependentRandomSampler.h"

namespace kaminpar::shm::sparsification {
template <>
double IndependentRandomSampler<EdgeWeight>::normalizationFactor(
    const CSRGraph &g, const StaticArray<EdgeWeight> &scores, EdgeID target
) {
  if (_noApprox)
    return exactNormalizationFactor(g, scores, target);
  else
    return approxNormalizationFactor(g, scores, target);
}
template <>
double IndependentRandomSampler<double>::normalizationFactor(
    const CSRGraph &g, const StaticArray<double> &scores, EdgeID target
) {
  return exactNormalizationFactor(g, scores, target);
}

template <>
double IndependentRandomSampler<EdgeWeight>::approxNormalizationFactor(
    const CSRGraph &g, const StaticArray<EdgeWeight> &scores, EdgeID target
) {
  EdgeID number_of_buckets = exponential_bucket(g.total_edge_weight()) + 1;
  std::vector<tbb::concurrent_vector<EdgeWeight>> expontial_buckets(number_of_buckets);
  StaticArray<EdgeWeight> buckets_score_prefixsum(number_of_buckets);
  StaticArray<EdgeID> buckets_size_prefixsum(number_of_buckets);
  tbb::parallel_for(static_cast<EdgeID>(0), g.m(), [&](EdgeID e) {
    EdgeWeight score = g.edge_weight(e);
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

  double factor = (target - (g.m() - buckets_size_prefixsum[bucket_index])) /
                  static_cast<double>(buckets_score_prefixsum[bucket_index]);
  return factor;
}

template <typename Score>
double IndependentRandomSampler<Score>::exactNormalizationFactor(
    const CSRGraph &g, const StaticArray<Score> &scores, EdgeID target
) {
  StaticArray<Score> sorted_scores(g.m() / 2);
  StaticArray<Score> prefix_sum(g.m() / 2);
  EdgeID end_of_sorted_scores = 0;
  utils::for_upward_edges(g, [&](EdgeID e) {
    sorted_scores[end_of_sorted_scores++] = static_cast<Score>(scores[e]);
  });
  tbb::parallel_sort(sorted_scores.begin(), sorted_scores.end());
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
  KASSERT(
      (index + 1 >= g.m() / 2 || expected_at_index(index + 1) <= target / 2) &&
          target / 2 <= expected_at_index(index),
      "binary search did not work: target/2=" << target / 2 << " is not in ["
                                              << expected_at_index(index + 1) << ", "
                                              << expected_at_index(index) << "]",
      assert::always
  );

  double factor = static_cast<double>((target / 2 - (g.m() / 2 - index))) / prefix_sum[index - 1];

  KASSERT(
      1.0 / sorted_scores[index] <= factor && factor <= 1.0 / sorted_scores[index - 1],
      "factor=" << factor << " not in interval [" << 1.0 / sorted_scores[index] << ", "
                << 1.0 / sorted_scores[index - 1] << "]",
      assert::always
  );
  return factor;
}

template class IndependentRandomSampler<EdgeWeight>;
template class IndependentRandomSampler<double>;
}; // namespace kaminpar::shm::sparsification
