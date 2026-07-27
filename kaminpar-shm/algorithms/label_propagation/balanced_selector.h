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
    EdgeWeight best_rating = 0;
    const BlockWeight from_overload = from_weight - _state.max_cluster_weight(from);
    BlockWeight best_overload = from_overload;
    UniformTieSet best_ties(ties);

    for (const auto [block, rating] : ratings.entries()) {
      const BlockWeight weight = _state.cluster_weight(block);
      const BlockWeight max_weight = _state.max_cluster_weight(block);
      const BlockWeight overload = weight - max_weight;
      const bool accepted =
          weight + u_weight <= max_weight || overload < from_overload || block == from;
      if (!accepted) {
        continue;
      }

      if (rating > best_rating) {
        best = block;
        best_rating = rating;
        best_overload = overload;
        best_ties.replace_with(block);
      } else if (rating == best_rating) {
        if (overload < best_overload) {
          best = block;
          best_overload = overload;
          best_ties.replace_with(block);
        } else if (overload == best_overload) {
          best_ties.add(block);
        }
      }
    }

    best = best_ties.select_or(best, random);
    best_ties.clear();
    return best;
  }

private:
  BalancedState &_state;
};

} // namespace kaminpar::shm::lp
