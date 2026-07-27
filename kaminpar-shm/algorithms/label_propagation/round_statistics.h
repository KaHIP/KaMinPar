/*******************************************************************************
 * Thread-local counters for a parallel label-propagation round.
 *
 * @file:   round_statistics.h
 * @author: Daniel Seemaier
 ******************************************************************************/
#pragma once

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm::lp {

struct RoundStats {
  NodeID moved = 0;
};

class RoundStatistics {
public:
  [[nodiscard]] RoundStats &local() {
    return _locals.local();
  }

  [[nodiscard]] RoundStats totals() {
    return _locals.combine([](const RoundStats lhs, const RoundStats rhs) {
      return RoundStats{lhs.moved + rhs.moved};
    });
  }

  void clear() {
    _locals.clear();
  }

private:
  tbb::enumerable_thread_specific<RoundStats> _locals;
};

} // namespace kaminpar::shm::lp
