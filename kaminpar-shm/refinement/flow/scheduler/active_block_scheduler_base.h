#pragma once

#include <chrono>
#include <cstddef>
#include <memory>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/flow_cutter/flow_cutter_algorithm.h"
#include "kaminpar-shm/refinement/flow/flow_network/border_region.h"
#include "kaminpar-shm/refinement/flow/flow_network/flow_network.h"
#include "kaminpar-shm/refinement/flow/flow_network/quotient_graph.h"

namespace kaminpar::shm {

struct ActiveBlockSchedulerStatistics {
  std::size_t num_searches;
  std::size_t num_local_improvements;
  std::size_t num_global_improvements;
  std::size_t num_move_conflicts;
  std::size_t num_imbalance_conflicts;
  std::size_t num_failed_imbalance_resolutions;
  std::size_t num_successful_imbalance_resolutions;
  double min_imbalance;
  double max_imbalance;
  double total_imbalance;

  void reset();
  void print() const;
};

class FlowRefiner {
public:
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = std::chrono::time_point<Clock>;

  using Move = FlowCutterAlgorithm::Move;
  using Result = FlowCutterAlgorithm::Result;

public:
  FlowRefiner(
      const PartitionContext &p_ctx,
      const TwowayFlowRefinementContext &f_ctx,
      bool run_sequentially,
      const QuotientGraph &q_graph,
      const PartitionedCSRGraph &p_graph,
      const CSRGraph &graph,
      const TimePoint &start_time
  );

  FlowRefiner(FlowRefiner &&) noexcept = default;
  FlowRefiner &operator=(FlowRefiner &&) noexcept = delete;

  FlowRefiner(const FlowRefiner &) = delete;
  FlowRefiner &operator=(const FlowRefiner &) = delete;

  [[nodiscard]] Result refine(BlockID block1, BlockID block2);

private:
  const PartitionedCSRGraph &_p_graph;
  bool _run_sequentially;

  BorderRegionConstructor _border_region_constructor;
  FlowNetworkConstructor _flow_network_constructor;
  std::unique_ptr<FlowCutterAlgorithm> _flow_cutter_algorithm;
};

} // namespace kaminpar::shm
