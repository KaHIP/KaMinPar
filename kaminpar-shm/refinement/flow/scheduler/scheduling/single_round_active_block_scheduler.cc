#include "kaminpar-shm/refinement/flow/scheduler/scheduling/single_round_active_block_scheduler.h"

#include <algorithm>

namespace kaminpar::shm {

SingleRoundActiveBlockScheduling::Scheduling SingleRoundActiveBlockScheduling::compute_scheduling(
    const QuotientGraph &quotient_graph, const std::span<const bool> active_blocks
) {
  return {compute_subround_scheduling(quotient_graph, active_blocks)};
}

SingleRoundActiveBlockScheduling::SubroundScheduling
SingleRoundActiveBlockScheduling::compute_subround_scheduling(
    const QuotientGraph &quotient_graph, std::span<const bool> active_blocks
) {
  ScalableVector<BlockPair> active_block_pairs;

  for (BlockID block2 = 1, k = quotient_graph.num_blocks(); block2 < k; ++block2) {
    for (BlockID block1 = 0; block1 < block2; ++block1) {
      if (quotient_graph.has_edge(block1, block2) &&
          (active_blocks[block1] || active_blocks[block2])) {
        active_block_pairs.emplace_back(block1, block2);
      }
    }
  }

  std::sort(
      active_block_pairs.begin(),
      active_block_pairs.end(),
      [&](const auto &pair1, const auto &pair2) {
        const QuotientGraph::Edge &edge1 = quotient_graph.edge(pair1.first, pair1.second);
        const QuotientGraph::Edge &edge2 = quotient_graph.edge(pair2.first, pair2.second);
        return edge1.total_gain > edge2.total_gain ||
               (edge1.total_gain == edge2.total_gain && (edge1.cut_weight > edge2.cut_weight));
      }
  );

  return active_block_pairs;
}

} // namespace kaminpar::shm
