#pragma once

#include <chrono>
#include <memory>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/flow_cutter/flow_cutter_algorithm.h"
#include "kaminpar-shm/refinement/flow/flow_network/quotient_graph.h"

namespace kaminpar::shm {

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
      const QuotientGraph &q_graph,
      const PartitionedCSRGraph &p_graph,
      const CSRGraph &graph,
      const TimePoint &start_time
  );

  FlowRefiner(FlowRefiner &&) noexcept = default;
  FlowRefiner &operator=(FlowRefiner &&) noexcept = delete;

  FlowRefiner(const FlowRefiner &) = delete;
  FlowRefiner &operator=(const FlowRefiner &) = delete;

  [[nodiscard]] Result refine(BlockID block1, BlockID block2, bool run_sequentially);

  void free();

private:
  const PartitionedCSRGraph &_p_graph;

  BorderRegionConstructor _border_region_constructor;
  FlowNetworkConstructor _flow_network_constructor;
  std::unique_ptr<FlowCutterAlgorithm> _flow_cutter_algorithm;
};

} // namespace kaminpar::shm
