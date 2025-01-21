/*******************************************************************************
 * Initial partitioner that invokes KaMinPar.
 *
 * @file:   kaminpar_initial_partitioner.cc
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 ******************************************************************************/
#include "kaminpar-dist/initial_partitioning/kaminpar_initial_partitioner.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/factories.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::dist {

shm::PartitionedGraph KaMinParInitialPartitioner::initial_partition(
    const shm::Graph &graph, const PartitionContext &p_ctx
) {
  if (graph.n() <= 1) {
    return {graph, p_ctx.k, StaticArray<BlockID>(graph.n())};
  }

  std::vector<shm::BlockWeight> max_block_weights(p_ctx.k);
  for (BlockID b = 0; b < p_ctx.k; ++b) {
    max_block_weights[b] = p_ctx.max_block_weight(b);
  }

  auto shm_ctx = _ctx.initial_partitioning.kaminpar;
  shm_ctx.refinement.lp.num_iterations = 1;
  shm_ctx.partition.setup(graph, std::move(max_block_weights));
  shm_ctx.compression.setup(graph);

  DISABLE_TIMERS();
  START_HEAP_PROFILER("KaMinPar");
  const bool was_quiet = Logger::is_quiet();
  Logger::set_quiet_mode(true);
  auto p_graph = shm::factory::create_partitioner(graph, shm_ctx)->partition();
  Logger::set_quiet_mode(was_quiet);
  STOP_HEAP_PROFILER();
  ENABLE_TIMERS();

  return p_graph;
}

} // namespace kaminpar::dist
