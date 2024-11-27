/*******************************************************************************
 * Deep multilevel graph partitioner that can use multiple v-cycles.
 *
 * @file:   vcycle_deep_multilevel.cc
 * @author: Daniel Seemaier
 * @date:   31.10.2024
 ******************************************************************************/
#include "kaminpar-shm/partitioning/deep/vcycle_deep_multilevel.h"

#include "kaminpar-shm/partitioning/deep/deep_multilevel.h"
#include "kaminpar-shm/partitioning/partition_utils.h"

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
  BlockID prev_k = 0;

  std::vector<BlockID> steps = _input_ctx.partitioning.vcycles;
  if (steps.empty() || steps.back() != _input_ctx.partition.k) {
    steps.push_back(_input_ctx.partition.k);
  }

  Context ctx = _input_ctx;
  ctx.partitioning.mode = PartitioningMode::DEEP;

  for (const BlockID current_k : steps) {
    {
      const int level = math::floor_log2(current_k);
      const BlockID expanded_blocks = current_k - (1 << level);

      std::vector<BlockWeight> max_block_weights(current_k);
      if (current_k == _input_ctx.partition.k) {
        for (BlockID b = 0; b < current_k; ++b) {
          max_block_weights[b] = _input_ctx.partition.max_block_weight(b);
        }
      } else {
        BlockID cur_begin = 0;
        for (BlockID b = 0; b < current_k; ++b) {
          const BlockID num_sub_blocks = [&] {
            if (b < 2 * expanded_blocks) {
              const BlockID next_k = std::min(math::ceil2(current_k), _input_ctx.partition.k);
              return partitioning::compute_final_k(b, next_k, _input_ctx.partition.k);
            } else {
              return partitioning::compute_final_k(
                  b - expanded_blocks, math::floor2(current_k), _input_ctx.partition.k
              );
            }
          }();

          LOG << num_sub_blocks;

          const BlockID cur_end = cur_begin + num_sub_blocks;
          max_block_weights[b] =
              _input_ctx.partition.total_unrelaxed_max_block_weights(cur_begin, cur_end);

          cur_begin = cur_end;
        }
      }
      ctx.partition.setup(_input_graph, std::move(max_block_weights));
    }

    if (communities.empty()) {
      ctx.partitioning.deep_initial_partitioning_mode =
          InitialPartitioningMode::ASYNCHRONOUS_PARALLEL;
    } else {
      ctx.partitioning.deep_initial_partitioning_mode = InitialPartitioningMode::COMMUNITIES;
    }

    DeepMultilevelPartitioner partitioner(_input_graph, ctx);
    partitioner.enable_metrics_output();
    if (prev_k > 0) {
      partitioner.use_communities(communities, prev_k);
    }
    PartitionedGraph p_graph = partitioner.partition();

    // Make sure that the restricted refinement option actually restricts nodes to their block
    KASSERT(
        !ctx.partitioning.restrict_vcycle_refinement || communities.empty() ||
            [&] {
              for (const NodeID u : p_graph.nodes()) {
                p_graph.adjacent_nodes(u, [&](const NodeID v) {
                  KASSERT(
                      p_graph.block(u) != p_graph.block(v) || communities[u] == communities[v],
                      "Node " << u << " and " << v
                              << " are in the same block, but in different communities",
                      assert::always
                  );
                });
              }
              return true;
            }(),
        "",
        assert::heavy
    );

    if (communities.empty()) {
      communities.resize(p_graph.n());
    }

    p_graph.pfor_nodes([&](const NodeID u) {
      communities[u] = p_graph.block(u);
      partition[u] = p_graph.block(u);
    });
    prev_k = p_graph.k();
  }

  return {_input_graph, _input_ctx.partition.k, std::move(partition)};
}

} // namespace kaminpar::shm
