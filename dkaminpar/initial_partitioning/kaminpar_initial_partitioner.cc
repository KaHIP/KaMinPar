/*******************************************************************************
 * @file:   kaminpar_initial_partitioner.cc
 *
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 * @brief:  Initial partitioner that invokes KaMinPar.
 ******************************************************************************/
#include "dkaminpar/initial_partitioning/kaminpar_initial_partitioner.h"

#include "kaminpar/partitioning_scheme/partitioning.h"
#include "kaminpar/utility/timer.h"

namespace dkaminpar {
shm::PartitionedGraph KaMinParInitialPartitioner::initial_partition(const shm::Graph &graph) {
  auto shm_ctx = _ctx.initial_partitioning.sequential;
  shm_ctx.refinement.lp.num_iterations = 1;
  shm_ctx.partition.k = _ctx.partition.k;
  shm_ctx.partition.epsilon = _ctx.partition.epsilon;
  shm_ctx.setup(graph);

  DISABLE_TIMERS();
  shm::Logger::set_quiet_mode(true);
  auto p_graph = shm::partitioning::partition(graph, shm_ctx);
  shm::Logger::set_quiet_mode(_ctx.quiet);
  ENABLE_TIMERS();

  return p_graph;
}
} // namespace dkaminpar