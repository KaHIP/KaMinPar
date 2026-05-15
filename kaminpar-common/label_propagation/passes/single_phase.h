/*******************************************************************************
 * Single-phase label propagation pass.
 *
 * @file:   single_phase.h
 ******************************************************************************/
#pragma once

#include "kaminpar-common/label_propagation/passes/basic.h"

namespace kaminpar::lp {

template <typename Kernel> struct SinglePhaseRatingMapAccessor {
  using RatingMap = typename Kernel::RatingMap;

  [[nodiscard]] static RatingMap &get(Kernel &kernel) {
    return kernel.workspace().rating.maps.local();
  }
};

template <typename Kernel, ActiveSetStrategy ActiveSet, TieBreakingStrategy TieBreaking>
class SinglePhasePass
    : public BasicPass<Kernel, ActiveSet, TieBreaking, SinglePhaseRatingMapAccessor<Kernel>> {
  using Base = BasicPass<Kernel, ActiveSet, TieBreaking, SinglePhaseRatingMapAccessor<Kernel>>;

public:
  using Base::Base;
};

} // namespace kaminpar::lp
