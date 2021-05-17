#include "partitioning_scheme/partitioning.h"

#include "partitioning_scheme/parallel_recursive_bisection.h"
#include "partitioning_scheme/parallel_simple_recursive_bisection.h"
#include "utility/timer.h"

namespace kaminpar::partitioning {
PartitionedGraph partition(const Graph &graph, const Context &ctx) {
  switch (ctx.partition.mode) {
    case PartitioningMode::DEEP: {
      START_TIMER(TIMER_PARTITIONING);
      START_TIMER(TIMER_ALLOCATION);
      ParallelRecursiveBisection rb{graph, ctx};
      STOP_TIMER();
      auto p_graph = rb.partition();
      STOP_TIMER();
      return p_graph;
    }

    case PartitioningMode::RB: {
      START_TIMER(TIMER_PARTITIONING);
      START_TIMER(TIMER_ALLOCATION);
      ParallelSimpleRecursiveBisection rb{graph, ctx};
      STOP_TIMER();
      auto p_graph = rb.partition();
      STOP_TIMER();
      return p_graph;
    }

    default:
      FATAL_ERROR << "Unsupported mode";
  }
  __builtin_unreachable();
}
} // namespace kaminpar::partitioning