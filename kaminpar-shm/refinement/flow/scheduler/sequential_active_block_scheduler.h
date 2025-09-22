#pragma once

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/flow_network/quotient_graph.h"
#include "kaminpar-shm/refinement/flow/scheduler/flow_refiner.h"
#include "kaminpar-shm/refinement/flow/scheduler/scheduling/active_block_scheduling.h"
#include "kaminpar-shm/refinement/flow/scheduler/scheduling/single_round_active_block_scheduler.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

class SequentialActiveBlockScheduler {
  SET_DEBUG(true);
  SET_STATISTICS(true);

  struct Statistics {
    std::size_t num_searches;
    std::size_t num_improvements;

    void reset() {
      num_searches = 0;
      num_improvements = 0;
    }

    void print() const {
      LOG_STATS << "Two-Way Flow Refiner:";
      LOG_STATS << "*  # num searches: " << num_searches;
      LOG_STATS << "*  # num improvements: " << num_improvements;
    }
  };

  using SubroundScheduling = ActiveBlockScheduling::SubroundScheduling;

  using Clock = FlowRefiner::Clock;
  using TimePoint = FlowRefiner::TimePoint;
  using Move = FlowRefiner::Move;
  using Result = FlowRefiner::Result;

public:
  SequentialActiveBlockScheduler(
      const ParallelContext &par_ctx, const TwowayFlowRefinementContext &f_ctx
  );

  SequentialActiveBlockScheduler(SequentialActiveBlockScheduler &&) noexcept = default;
  SequentialActiveBlockScheduler &operator=(SequentialActiveBlockScheduler &&) noexcept = delete;

  SequentialActiveBlockScheduler(const SequentialActiveBlockScheduler &) = delete;
  SequentialActiveBlockScheduler &operator=(const SequentialActiveBlockScheduler &) = delete;

  bool refine(PartitionedCSRGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx);

private:
  void apply_moves(std::span<const Move> moves);

private:
  const ParallelContext &_par_ctx;
  const TwowayFlowRefinementContext &_f_ctx;
  Statistics _stats;

  PartitionedCSRGraph *_p_graph;
  const CSRGraph *_graph;

  StaticArray<bool> _active_blocks;
  SingleRoundActiveBlockScheduling _active_block_scheduling;
  ScalableVector<QuotientGraph::GraphEdge> _new_cut_edges;
};

} // namespace kaminpar::shm
