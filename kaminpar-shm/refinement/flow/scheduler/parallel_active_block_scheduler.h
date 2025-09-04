#pragma once

#include <memory>
#include <mutex>
#include <span>
#include <utility>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/flow_network/quotient_graph.h"
#include "kaminpar-shm/refinement/flow/scheduler/active_block_scheduler_base.h"
#include "kaminpar-shm/refinement/flow/scheduler/scheduling/active_block_scheduling.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

class ParallelActiveBlockScheduler {
  SET_DEBUG(true);
  SET_STATISTICS(true);

  using Scheduling = ActiveBlockScheduling::Scheduling;
  using QuotientCutEdges = ScalableVector<QuotientGraph::GraphEdge>;

  using Clock = FlowRefiner::Clock;
  using TimePoint = FlowRefiner::TimePoint;
  using Move = FlowRefiner::Move;

  enum class MoveResult {
    IMBALANCE_CONFLICT,
    NEGATIVE_GAIN,
    SUCCESS,
  };

  struct MoveAttempt {
    MoveResult kind;
    double imbalance;
    EdgeWeight cut_value;
    EdgeWeight gain;

    MoveAttempt(MoveResult kind, double imbalance)
        : kind(kind),
          imbalance(imbalance),
          cut_value(0),
          gain(0) {}

    MoveAttempt(MoveResult kind, EdgeWeight cut_value, EdgeWeight gain)
        : kind(kind),
          imbalance(0.0),
          cut_value(cut_value),
          gain(gain) {}
  };

public:
  ParallelActiveBlockScheduler(const TwowayFlowRefinementContext &f_ctx);

  ParallelActiveBlockScheduler(ParallelActiveBlockScheduler &&) noexcept = delete;
  ParallelActiveBlockScheduler &operator=(ParallelActiveBlockScheduler &&) noexcept = delete;

  ParallelActiveBlockScheduler(const ParallelActiveBlockScheduler &) = delete;
  ParallelActiveBlockScheduler &operator=(const ParallelActiveBlockScheduler &) = delete;

  bool refine(PartitionedCSRGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx);

private:
  void commit_moves(
      EdgeWeight &cut_value,
      EdgeWeight gain,
      BlockID block1,
      BlockID block2,
      std::span<const Move> moves,
      QuotientGraph &quotient_graph,
      QuotientCutEdges &new_cut_edges
  );

  void apply_moves(std::span<const Move> moves, QuotientCutEdges &new_cut_edges);

  MoveAttempt commit_moves_if_feasible(
      EdgeWeight &cut_value,
      BlockID block1,
      BlockID block2,
      std::span<Move> moves,
      QuotientCutEdges &new_cut_edges
  );

  std::pair<bool, double> is_feasible_move_sequence(std::span<Move> moves);

  EdgeWeight atomic_apply_moves(std::span<const Move> moves, QuotientCutEdges &new_cut_edges);

  void revert_moves(std::span<const Move> moves);

private:
  const TwowayFlowRefinementContext &_f_ctx;
  ActiveBlockSchedulerStatistics _stats;

  PartitionedCSRGraph *_p_graph;
  const CSRGraph *_graph;
  const PartitionContext *_p_ctx;

  StaticArray<bool> _active_blocks;
  std::unique_ptr<ActiveBlockScheduling> _active_block_scheduling;

  std::mutex _apply_moves_mutex;
  StaticArray<BlockWeight> _block_weight_delta;
};

} // namespace kaminpar::shm
