#pragma once

#include <span>

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/flow_network/quotient_graph.h"
#include "kaminpar-shm/refinement/flow/scheduler/scheduling/active_block_scheduling.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

class MatchingBasedActiveBlockScheduling : public ActiveBlockScheduling {
public:
  MatchingBasedActiveBlockScheduling(const FlowSchedulerContext &ctx) : _ctx(ctx) {}

  [[nodiscard]] Scheduling compute_scheduling(
      const QuotientGraph &quotient_graph, std::span<const bool> active_blocks, std::size_t round
  ) override;

private:
  const FlowSchedulerContext &_ctx;

  ScalableVector<BlockID> _active_blocks;
  StaticArray<BlockID> _active_block_degrees;
  ScalableVector<ScalableVector<BlockID>> _adjacent_active_blocks;
  StaticArray<bool> _matched;
  StaticArray<bool> _scheduled;
};

} // namespace kaminpar::shm
