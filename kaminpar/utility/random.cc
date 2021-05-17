#include "utility/random.h"

namespace kaminpar {
Randomize &Randomize::instance() {
  thread_local static Randomize instance;
  return instance;
}

int Randomize::seed = 0;
}