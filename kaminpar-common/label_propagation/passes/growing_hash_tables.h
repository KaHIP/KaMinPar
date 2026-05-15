/*******************************************************************************
 * Growing-hash-table label propagation pass.
 *
 * @file:   growing_hash_tables.h
 ******************************************************************************/
#pragma once

#include "kaminpar-common/label_propagation/passes/basic.h"

namespace kaminpar::lp {

template <typename Kernel> struct GrowingRatingMapAccessor {
  using RatingMap = typename Kernel::GrowingRatingMap;

  [[nodiscard]] static RatingMap &get(Kernel &kernel) {
    return kernel.workspace().rating.growing_maps.local();
  }
};

template <typename Kernel, ActiveSetStrategy ActiveSet, TieBreakingStrategy TieBreaking>
class GrowingHashTablePass
    : public BasicPass<Kernel, ActiveSet, TieBreaking, GrowingRatingMapAccessor<Kernel>> {
  using Base = BasicPass<Kernel, ActiveSet, TieBreaking, GrowingRatingMapAccessor<Kernel>>;

public:
  using Base::Base;
};

} // namespace kaminpar::lp
