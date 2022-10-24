/*******************************************************************************
 * @file:   partitioning.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:
 ******************************************************************************/
#include "kaminpar/partitioning/partitioning.h"

#include "kaminpar/partitioning/parallel_recursive_bisection.h"
#include "kaminpar/partitioning/parallel_simple_recursive_bisection.h"

#include "common/logger.h"
#include "common/timer.h"

namespace kaminpar::shm::partitioning {
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
} // namespace kaminpar::shm::partitioning