/*******************************************************************************
 * Deep multilevel graph partitioner that can use multiple v-cycles.
 *
 * @file:   vcycle_deep_multilevel.cc
 * @author: Daniel Seemaier
 * @date:   31.10.2024
 ******************************************************************************/
#include "kaminpar-shm/partitioning/deep/vcycle_deep_multilevel.h"

#include "kaminpar-shm/partitioning/deep/deep_multilevel.h"

namespace kaminpar::shm {

namespace {

SET_DEBUG(false);

} // namespace

VcycleDeepMultilevelPartitioner::VcycleDeepMultilevelPartitioner(
    const Graph &input_graph, const Context &input_ctx
)
    : _input_graph(input_graph),
      _input_ctx(input_ctx) {}

PartitionedGraph VcycleDeepMultilevelPartitioner::partition() {
  // Need two copies of the same thing since BlockID and NodeID could be different types
  StaticArray<NodeID> communities;
  StaticArray<BlockID> partition(_input_graph.n());

  std::vector<BlockID> steps = _input_ctx.partitioning.vcycles;
  if (steps.empty() || steps.back() != _input_ctx.partition.k) {
    steps.push_back(_input_ctx.partition.k);
  }

  Context ctx = _input_ctx;
  ctx.partitioning.mode = PartitioningMode::DEEP;

  for (const BlockID k : steps) {
    ctx.partition.k = k;
    ctx.partition.setup(_input_graph);

    if (communities.empty()) {
      ctx.partitioning.deep_initial_partitioning_mode =
          InitialPartitioningMode::ASYNCHRONOUS_PARALLEL;
    } else {
      ctx.partitioning.deep_initial_partitioning_mode = InitialPartitioningMode::COMMUNITIES;
    }

    DeepMultilevelPartitioner partitioner(_input_graph, ctx);
    partitioner.use_communities(communities);
    PartitionedGraph p_graph = partitioner.partition();

    if (communities.empty()) {
      communities.resize(p_graph.n());
    }

    p_graph.pfor_nodes([&](const NodeID u) {
      communities[u] = p_graph.block(u);
      partition[u] = p_graph.block(u);
    });
  }

  return {_input_graph, _input_ctx.partition.k, std::move(partition)};
}

} // namespace kaminpar::shm
