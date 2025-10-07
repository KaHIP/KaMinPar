#pragma once

#include <span>

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/flow_network/quotient_graph.h"
#include "kaminpar-shm/refinement/flow/scheduler/scheduling/active_block_scheduling.h"

namespace kaminpar::shm {

class SingleRoundActiveBlockScheduling : public ActiveBlockScheduling {
public:
  SingleRoundActiveBlockScheduling(const FlowSchedulerContext &ctx) : _ctx(ctx) {}

  [[nodiscard]] Scheduling compute_scheduling(
      const QuotientGraph &quotient_graph, std::span<const bool> active_blocks, std::size_t round
  ) override;

  [[nodiscard]] SubroundScheduling compute_subround_scheduling(
      const QuotientGraph &quotient_graph,
      std::span<const bool> active_blocks,
      const std::size_t round
  );

private:
  const FlowSchedulerContext &_ctx;
};

} // namespace kaminpar::shm
