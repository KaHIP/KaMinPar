/*******************************************************************************
 * Value types exchanged by label-propagation components.
 *
 * @file:   move.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

namespace kaminpar::shm::lp {

struct MoveResult {
  bool moved = false;
  bool emptied_cluster = false;
};

} // namespace kaminpar::shm::lp
