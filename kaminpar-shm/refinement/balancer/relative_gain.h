/*******************************************************************************
 * Relative gain computation.
 *
 * @file:   relative_gain.h
 * @author: Daniel Seemaier
 * @date:   26.08.2025
 ******************************************************************************/
#pragma once

#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {

inline float compute_relative_gain(const EdgeWeight gain, const NodeWeight weight) {
  return gain > 0 ? 1.0f * gain * weight : 1.0f * gain / weight;
}

} // namespace kaminpar::shm
