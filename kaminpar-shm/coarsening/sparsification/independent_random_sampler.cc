#include "kaminpar-shm/coarsening/sparsification/independent_random_sampler.h"

#include <ranges>

#include "kaminpar-shm/coarsening/sparsification/sparsification_utils.h"
#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm::sparsification {

template <typename Score>
StaticArray<EdgeWeight>
IndependentRandomSampler<Score>::sample(const CSRGraph &g, EdgeID target_edge_amount) {
  auto scores = this->_score_function->scores(g);
  double factor = normalizationFactor(g, scores, target_edge_amount);

  StaticArray<EdgeWeight> sample(g.m(), 0);
  utils::parallel_for_upward_edges(g, [&](EdgeID e) {
    sample[e] = Random::instance().random_bool(factor * scores[e]) ? g.edge_weight(e) : 0;
  });
  return sample;
}

template <>
double IndependentRandomSampler<EdgeWeight>::normalizationFactor(
    const CSRGraph &g, const StaticArray<EdgeWeight> &scores, EdgeID target
) {
  if (_no_approx) {
    return exactNormalizationFactor(g, scores, target);
  } else {
    return approxNormalizationFactor(g, scores, target);
  }
}

template <>
double IndependentRandomSampler<EdgeID>::normalizationFactor(
    const CSRGraph &g, const StaticArray<EdgeID> &scores, EdgeID target
) {
  if (_no_approx) {
    return exactNormalizationFactor(g, scores, target);
  } else {
    return approxNormalizationFactor(g, scores, target);
  }
}

template <>
double IndependentRandomSampler<double>::normalizationFactor(
    const CSRGraph &g, const StaticArray<double> &scores, EdgeID target
) {
  return exactNormalizationFactor(g, scores, target);
}

template <typename Score>
double IndependentRandomSampler<Score>::approxNormalizationFactor(
    const CSRGraph &g, const StaticArray<Score> &scores, EdgeID target
) {
  // The i-th bucket contains scores in [2^i, 2^(i+1))
  Score max_score = *std::max_element(scores.begin(), scores.end());
  EdgeID number_of_buckets = exponential_bucket(max_score) + 1;
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

  auto max_edges_with_factor_in_bucket = [&](EdgeID bucket_index) {
    // s = smallest possible score in bucket = 2^bucket_index
    // #{e in Edges : s <= scores[e]} + 1/s scores{e in Edges : scores[e] < s}
    if (bucket_index > 1)
      return g.m() - buckets_size_prefixsum[bucket_index - 1] +
             1.0 / (1 << bucket_index) * buckets_score_prefixsum[bucket_index - 1];
    else
      return static_cast<double>(g.m());
  };
  EdgeID bucket_index = number_of_buckets - 1;
  while (target > max_edges_with_factor_in_bucket(bucket_index))
    bucket_index -= 1;

  double factor = (target - (g.m() - buckets_size_prefixsum[bucket_index])) /
                  static_cast<double>(buckets_score_prefixsum[bucket_index]);
  return factor;
}

template <>
double IndependentRandomSampler<double>::approxNormalizationFactor(
    const CSRGraph & /* g */, const StaticArray<double> & /* scores */, EdgeID /* target */
) {
  throw std::logic_error(
      "no implementation for of approxNormalizationFactor exists for Score=double."
  );
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
      [&](EdgeID t, NodeID i) { return t <= expected_at_index(i); }
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

  return factor;
}

} // namespace kaminpar::shm::sparsification
