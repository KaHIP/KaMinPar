/*******************************************************************************
 * @file:   kaminpar_initial_partitioner.cc
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 * @brief:  Initial partitioner that invokes KaMinPar.
 ******************************************************************************/
#include "dkaminpar/initial_partitioning/kaminpar_initial_partitioner.h"

#include "kaminpar/partitioning_scheme/partitioning.h"

#include "common/logger.h"
#include "common/timer.h"

namespace kaminpar::dist {
shm::PartitionedGraph
KaMinParInitialPartitioner::initial_partition(const shm::Graph& graph, const PartitionContext& p_ctx) {
    auto shm_ctx                         = _ctx.initial_partitioning.kaminpar;
    shm_ctx.refinement.lp.num_iterations = 1;
    shm_ctx.partition.k                  = p_ctx.k;
    shm_ctx.partition.epsilon            = p_ctx.epsilon;
    shm_ctx.setup(graph);

    DISABLE_TIMERS();
    Logger::set_quiet_mode(true);
    auto p_graph = shm::partitioning::partition(graph, shm_ctx);
    Logger::set_quiet_mode(_ctx.quiet);
    ENABLE_TIMERS();

    return p_graph;
}
} // namespace kaminpar::dist
