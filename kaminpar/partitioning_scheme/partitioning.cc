/*******************************************************************************
 * @file:   partitioning.cc
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:
 ******************************************************************************/
#include "kaminpar/partitioning_scheme/partitioning.h"

#include "kaminpar/partitioning_scheme/parallel_recursive_bisection.h"
#include "kaminpar/partitioning_scheme/parallel_simple_recursive_bisection.h"
#include "kaminpar/utils/timer.h"

namespace kaminpar::partitioning {
PartitionedGraph partition(const Graph& graph, const Context& ctx) {
    switch (ctx.partition.mode) {
        case PartitioningMode::DEEP: {
            START_TIMER("Partitioning");
            START_TIMER("Allocation");
            ParallelRecursiveBisection rb{graph, ctx};
            STOP_TIMER();
            auto p_graph = rb.partition();
            STOP_TIMER();
            return p_graph;
        }

        case PartitioningMode::RB: {
            START_TIMER("Partitioning");
            START_TIMER("Allocation");
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