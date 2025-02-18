/*******************************************************************************
 * Interface for initial refinement algorithms.
 *
 * @file:   initial_refiner.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/initial_partitioning/initial_refiner.h"

#include "kaminpar-shm/initial_partitioning/initial_fm_refiner.h"
#include "kaminpar-shm/initial_partitioning/initial_noop_refiner.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {

std::unique_ptr<InitialRefiner> create_initial_refiner(const InitialRefinementContext &r_ctx) {
  if (r_ctx.disabled) {
    return std::make_unique<InitialNoopRefiner>();
  }

  switch (r_ctx.stopping_rule) {
  case FMStoppingRule::ADAPTIVE:
    return std::make_unique<InitialSimple2WayFM>(r_ctx);

  case FMStoppingRule::SIMPLE:
    return std::make_unique<InitialAdaptive2WayFM>(r_ctx);
  }

  __builtin_unreachable();
}

} // namespace kaminpar::shm
