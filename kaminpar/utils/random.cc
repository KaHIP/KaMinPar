/*******************************************************************************
 * @file:   random.cc
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:  Helper class for randomization.
 ******************************************************************************/
#include "kaminpar/utils/random.h"

namespace kaminpar {
Randomize& Randomize::instance() {
    thread_local static Randomize instance;
    return instance;
}

int Randomize::seed = 0;
} // namespace kaminpar