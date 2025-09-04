#pragma once

#include <span>

#include "kaminpar-shm/refinement/flow/flow_network/quotient_graph.h"
#include "kaminpar-shm/refinement/flow/scheduler/scheduling/active_block_scheduling.h"

namespace kaminpar::shm {

class SingleRoundActiveBlockScheduling : public ActiveBlockScheduling {
public:
  [[nodiscard]] Scheduling compute_scheduling(
      const QuotientGraph &quotient_graph, std::span<const bool> active_blocks
  ) override;

  [[nodiscard]] SubroundScheduling compute_subround_scheduling(
      const QuotientGraph &quotient_graph, std::span<const bool> active_blocks
  );
};

} // namespace kaminpar::shm
