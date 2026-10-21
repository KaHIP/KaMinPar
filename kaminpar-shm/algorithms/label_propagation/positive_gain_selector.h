/*******************************************************************************
 * Selects a maximum-rating target with strictly positive gain.
 *
 * @file:   positive_gain_selector.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <limits>
#include <type_traits>
#include <utility>

#include "kaminpar-shm/algorithms/label_propagation/uniform_tie_set.h"

#include "kaminpar-common/inline.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm::lp {

template <typename Gain, typename ID, typename Entries, typename TieContainer>
KAMINPAR_INLINE std::pair<ID, Gain> select_best_positive_gain(
    const ID from, Entries &&entries, Random &random, TieContainer &tie_storage
) {
  static_assert(std::is_signed_v<Gain>);

  Gain source_rating = 0;
  ID best_target = from;
  Gain best_rating = std::numeric_limits<Gain>::lowest();
  bool found_target = false;
  UniformTieSet ties(tie_storage);

  for (const auto [target, rating] : entries) {
    if (target == from) {
      source_rating = rating;
      continue;
    }

    if (!found_target || rating > best_rating) {
      best_target = target;
      best_rating = rating;
      found_target = true;
      ties.clear();
    } else if (rating == best_rating) {
      if (tie_storage.empty()) {
        ties.add(best_target);
      }
      ties.add(target);
    }
  }

  Gain best_gain = 0;
  if (found_target) {
    const Gain candidate_gain = best_rating - source_rating;
    if (candidate_gain > 0) {
      best_gain = candidate_gain;
      best_target = ties.select_or(best_target, random);
    } else {
      best_target = from;
    }
  }
  ties.clear();

  return {best_target, best_gain};
}

} // namespace kaminpar::shm::lp
