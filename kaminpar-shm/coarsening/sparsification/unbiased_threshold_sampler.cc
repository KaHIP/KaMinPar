#include "kaminpar-shm/coarsening/sparsification/unbiased_threshold_sampler.h"

#include <ranges>
#include <vector>

#include "kaminpar-shm/coarsening/sparsification/sparsification_utils.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm::sparsification {
StaticArray<EdgeWeight>
UnbiasedThesholdSampler::sample(const CSRGraph &g, EdgeID target_edge_amount) {
  std::vector nodes = {0, 3, 5, 7, 8};
  std::vector edges = {1, 2, 4, 0, 2, 0, 1, 0};
  std::vector edge_weights = {2, 7, 20, 2, 1, 7, 1, 20};
  CSRGraph test(
      StaticArray<EdgeID>(nodes.begin(), nodes.end()),
      StaticArray<NodeID>(edges.begin(), edges.end()),
      {},
      StaticArray<EdgeWeight>(edge_weights.begin(), edge_weights.end())
  );
  auto test_result = find_threshold(test, 4);
  KASSERT(test_result == 10);

  double threshold = find_threshold(g, target_edge_amount);

  StaticArray<EdgeWeight> sample(g.m(), 0);

  utils::for_upward_edges(g, [&](EdgeID e) {
    if (threshold <= g.edge_weight(e)) {
      sample[e] = g.edge_weight(e);
    } else if (Random::instance().random_bool(static_cast<double>(g.edge_weight(e)) / threshold)) {
      KASSERT(static_cast<double>(g.edge_weight(e)) / threshold < 1, assert::always);
      // randomized rounding
      sample[e] = Random::instance().random_bool(threshold - std::floor(threshold))
                      ? std::ceil(threshold)
                      : std::floor(threshold);
    } else {
      KASSERT(false, assert::always);
    }
  });

  return sample;
}

double UnbiasedThesholdSampler::find_threshold(const CSRGraph &g, EdgeID target_edge_amount) {
  StaticArray<EdgeWeight> sorted_weights(g.m() / 2);
  EdgeID i = 0;
  utils::for_upward_edges(g, [&](EdgeID e) { sorted_weights[i++] = g.edge_weight(e); });
  std::sort(sorted_weights.begin(), sorted_weights.end());

  std::vector<std::pair<EdgeWeight, EdgeID>> deduplicated_sorted_weights_with_number = {
      std::pair(sorted_weights[0], 0)
  };
  for (auto w : sorted_weights) {
    auto [previous_weight, number] = deduplicated_sorted_weights_with_number.back();
    if (previous_weight == w) {
      deduplicated_sorted_weights_with_number.pop_back();
      deduplicated_sorted_weights_with_number.emplace_back(w, number + 1);
    } else {
      deduplicated_sorted_weights_with_number.emplace_back(w, 1);
    }
  }

  std::vector<EdgeID> number_of_lighter_edges = {0};
  std::vector<double> weight_of_lighter_edges = {0};
  for (auto [w, number] : deduplicated_sorted_weights_with_number) {
    number_of_lighter_edges.push_back(number_of_lighter_edges.back() + number);
    weight_of_lighter_edges.push_back(weight_of_lighter_edges.back() + w * number);
  }

  auto possible_indices = std::ranges::iota_view(
      static_cast<EdgeID>(0), static_cast<EdgeID>(deduplicated_sorted_weights_with_number.size())
  );
  EdgeID index = *std::upper_bound(
      possible_indices.begin(),
      possible_indices.end(),
      target_edge_amount / 2,
      [&](EdgeID target, EdgeID index) {
        KASSERT(target_edge_amount / 2 == target, "tmp", assert::always);
        // > since decending
        return target > g.m() / 2 - number_of_lighter_edges[index] +
                            weight_of_lighter_edges[index] /
                                std::get<0>(deduplicated_sorted_weights_with_number[index]);
      }
  );

  return weight_of_lighter_edges[index] /
         (target_edge_amount / 2 - g.m() / 2 + number_of_lighter_edges[index]);
}

} // namespace kaminpar::shm::sparsification
