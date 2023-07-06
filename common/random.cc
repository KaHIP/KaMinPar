/*******************************************************************************
 * Thread-local singleton for PRNG.
 *
 * @file:   random.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "common/random.h"

namespace kaminpar {
Random &Random::instance() {
  thread_local static Random instance;
  return instance;
}

int Random::seed = 0;
} // namespace kaminpar
