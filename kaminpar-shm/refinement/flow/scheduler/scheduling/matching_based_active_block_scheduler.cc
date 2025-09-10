#include "kaminpar-shm/refinement/flow/scheduler/scheduling/matching_based_active_block_scheduler.h"

#include <algorithm>
#include <unordered_set>
#include <utility>

#include "kaminpar-common/assert.h"

namespace kaminpar::shm {

MatchingBasedActiveBlockScheduling::Scheduling
MatchingBasedActiveBlockScheduling::compute_scheduling(
    const QuotientGraph &quotient_graph,
    const std::span<const bool> active_blocks,
    [[maybe_unused]] const std::size_t round
) {
  const BlockID k = quotient_graph.num_blocks();

  if (_active_block_degrees.size() < k) {
    _active_block_degrees.resize(k, static_array::noinit);
  }
  if (_adjacent_active_blocks.size() < k) {
    _adjacent_active_blocks.resize(k);
  }
  if (_matched.size() < k) {
    _matched.resize(k);
  }
  if (_scheduled.size() < k * k) {
    _scheduled.resize(k * k);
  }
  std::fill_n(_scheduled.begin(), k * k, false);

  for (BlockID block = 0; block < k; ++block) {
    _active_block_degrees[block] = 0;
    _adjacent_active_blocks[block].clear();
  }

  const auto skip_block_pair = [&](const BlockID block1, const BlockID block2) {
    const QuotientGraph::Edge &quotient_edge = quotient_graph.edge(block1, block2);

    if (_ctx.skip_small_cuts && quotient_edge.cut_weight < _ctx.small_cut_threshold) {
      return true;
    }

    if (_ctx.skip_unpromising_cuts && round > 1 && quotient_edge.total_gain == 0) {
      return true;
    }

    return false;
  };

  BlockID num_active_block_pairs = 0;
  for (BlockID block2 = 1; block2 < k; ++block2) {
    for (BlockID block1 = 0; block1 < block2; ++block1) {
      if (quotient_graph.has_edge(block1, block2) &&
          (active_blocks[block1] || active_blocks[block2]) && !skip_block_pair(block1, block2)) {
        num_active_block_pairs += 1;

        _active_block_degrees[block1] += 1;
        _active_block_degrees[block2] += 1;

        _adjacent_active_blocks[block1].push_back(block2);
        _adjacent_active_blocks[block2].push_back(block1);
      }
    }
  }

  for (BlockID block = 0; block < k; ++block) {
    if (_active_block_degrees[block] > 0) {
      _active_blocks.push_back(block);
    }
  }

  const auto comparator = [&](const auto &b1, const auto &b2) {
    const BlockID b1_degree = _active_block_degrees[b1];
    const BlockID b2_degree = _active_block_degrees[b2];
    return b1_degree > b2_degree || (b1_degree == b2_degree && b1 < b2);
  };

  const auto fix_ordering = [&](const BlockID block) {
    std::size_t i = 0;
    while (i < _active_blocks.size() && _active_blocks[i] != block) {
      i += 1;
    }
    KASSERT(_active_blocks[i] == block);

    std::size_t j = i;
    while (j + 1 < _active_blocks.size() && !comparator(block, _active_blocks[j + 1])) {
      j += 1;
    }

    std::swap(_active_blocks[i], _active_blocks[j]);

    if (_active_block_degrees[block] == 0) {
      KASSERT(_active_blocks.back() == block);
      _active_blocks.pop_back();
    }
  };

  std::sort(_active_blocks.begin(), _active_blocks.end(), comparator);

  Scheduling scheduling;
  while (num_active_block_pairs > 0) {
    ScalableVector<BlockPair> active_block_pairs;

    std::fill_n(_matched.begin(), k, false);
    for (std::size_t i = 0; i < _active_blocks.size(); ++i) {
      const BlockID block = _active_blocks[i];
      if (_matched[block]) {
        continue;
      }

      for (const BlockID adjacent_block : _adjacent_active_blocks[block]) {
        if (_matched[adjacent_block]) {
          continue;
        }

        const BlockID smaller_block = std::min(block, adjacent_block);
        const BlockID larger_block = std::max(block, adjacent_block);
        if (_scheduled[smaller_block * k + larger_block]) {
          continue;
        }

        num_active_block_pairs -= 1;
        active_block_pairs.emplace_back(smaller_block, larger_block);

        _active_block_degrees[block] -= 1;
        fix_ordering(block);

        _active_block_degrees[adjacent_block] -= 1;
        fix_ordering(adjacent_block);

        _matched[block] = true;
        _matched[adjacent_block] = true;

        _scheduled[smaller_block * k + larger_block] = true;
        break;
      }
    }

    scheduling.push_back(std::move(active_block_pairs));
  }

  KASSERT(_active_blocks.empty());
  KASSERT(
      [&] {
        std::unordered_set<BlockID> matched_blocks;

        for (auto &active_block_pairs : scheduling) {
          matched_blocks.clear();

          for (auto &[block_0, block_1] : active_block_pairs) {
            if (matched_blocks.contains(block_0) || matched_blocks.contains(block_1)) {
              return false;
            }

            matched_blocks.insert(block_0);
            matched_blocks.insert(block_1);
          }
        }

        return true;
      }(),
      "Failed to compute a valid matching in one of the scheduling rounds",
      assert::heavy
  );
  KASSERT(
      [&] {
        for (BlockID block2 = 1; block2 < k; ++block2) {
          for (BlockID block1 = 0; block1 < block2; ++block1) {
            if (quotient_graph.has_edge(block1, block2) &&
                (active_blocks[block1] || active_blocks[block2])) {
              if (!_scheduled[block1 * k + block2]) {
                return false;
              }
            }
          }
        }

        return true;
      }(),
      "Not all active block pairs were scheduled",
      assert::heavy
  );

  return scheduling;
}

} // namespace kaminpar::shm
