/*******************************************************************************
 * Initial partitioner that assigns nodes to blocks randomly.
 *
 * @file:   random_initial_partitioner.cc
 * @author: Daniel Seemaier
 * @date:   06.11.2021
 ******************************************************************************/
#include "dkaminpar/initial_partitioning/random_initial_partitioner.h"

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "dkaminpar/context.h"
#include "dkaminpar/dkaminpar.h"

#include "common/datastructures/static_array.h"
#include "common/random.h"

namespace kaminpar::dist {
shm::PartitionedGraph RandomInitialPartitioner::initial_partition(
    const shm::Graph &graph, const PartitionContext &p_ctx
) {
  StaticArray<BlockID> partition(graph.n());
  std::vector<BlockID> final_k(graph.n(), 1);

  tbb::parallel_for(tbb::blocked_range<NodeID>(0, graph.n()), [&](const auto &r) {
    auto &rand = Random::instance();
    for (NodeID u = r.begin(); u != r.end(); ++u) {
      partition[u] = rand.random_index(0, p_ctx.k);
    }
  });

  return {graph, p_ctx.k, std::move(partition), std::move(final_k)};
}
} // namespace kaminpar::dist
