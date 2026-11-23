/*******************************************************************************
 * Uniform block selection for balanced label-propagation refinement.
 *
 * @file:   balanced_selector.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <type_traits>

#include "kaminpar-shm/algorithms/label_propagation/balanced_state.h"
#include "kaminpar-shm/algorithms/label_propagation/uniform_tie_set.h"

#include "kaminpar-common/random.h"

namespace kaminpar::shm::lp {

class BalancedSelector {
public:
  explicit BalancedSelector(BalancedState &state) : _state(state) {}

  template <typename Map, typename TieContainer>
  [[nodiscard]] BlockID select(
      const BlockID from,
      const NodeWeight u_weight,
      Map &ratings,
      Random &random,
      TieContainer &ties
  ) const {
    static_assert(std::is_signed_v<NodeWeight>);

    const BlockWeight from_weight = _state.cluster_weight(from);
    if (from_weight - u_weight < _state.min_cluster_weight(from)) {
      return from;
    }

    BlockID best = from;
    BlockID best_second = from;
    std::size_t best_count = 0;
    EdgeWeight best_rating = 0;
    const BlockWeight from_overload = from_weight - _state.max_cluster_weight(from);
    BlockWeight best_overload = from_overload;

    for (const auto [block, rating] : ratings.entries()) {
      if (rating < best_rating) {
        continue;
      }

      const BlockWeight weight = _state.cluster_weight(block);
      const BlockWeight max_weight = _state.max_cluster_weight(block);
      const BlockWeight overload = weight - max_weight;
      const bool accepted =
          block == from || weight + u_weight <= max_weight || overload < from_overload;
      if (!accepted) {
        continue;
      }

      if (rating > best_rating) {
        best = block;
        best_rating = rating;
        best_overload = overload;
        best_count = 1;
      } else {
        if (overload < best_overload) {
          best = block;
          best_overload = overload;
          best_count = 1;
        } else if (overload == best_overload) {
          if (best_count == 0) {
            best = block;
            best_count = 1;
          } else if (best_count == 1) {
            best_second = block;
            best_count = 2;
          } else {
            if (best_count == 2) {
              ties.clear();
              ties.push_back(best);
              ties.push_back(best_second);
            }
            ties.push_back(block);
            ++best_count;
          }
        }
      }
    }

    if (best_count == 2) {
      if (random.random_bool()) {
        best = best_second;
      }
    } else if (best_count > 2) {
      best = ties[random.random_index(0, best_count)];
    }
    return best;
  }

private:
  BalancedState &_state;
};

} // namespace kaminpar::shm::lp
