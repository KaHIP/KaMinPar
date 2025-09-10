#pragma once

#include <span>
#include <utility>

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/flow_network/quotient_graph.h"

#include "kaminpar-common/datastructures/scalable_vector.h"

namespace kaminpar::shm {

class ActiveBlockScheduling {
public:
  using BlockPair = std::pair<BlockID, BlockID>;
  using SubroundScheduling = ScalableVector<BlockPair>;
  using Scheduling = ScalableVector<SubroundScheduling>;

  [[nodiscard]] virtual Scheduling compute_scheduling(
      const QuotientGraph &quotient_graph, std::span<const bool> active_blocks, std::size_t round
  ) = 0;
};

} // namespace kaminpar::shm
