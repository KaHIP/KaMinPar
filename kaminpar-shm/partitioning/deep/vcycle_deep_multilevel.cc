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

SET_DEBUG(true);

std::vector<BlockWeight> compute_max_block_weights(
    const BlockID current_k,
    const PartitionContext &input_p_ctx,
    const std::vector<BlockWeight> &next_block_weights
) {
  const int level = math::floor_log2(current_k);
  const BlockID expanded_blocks = current_k - (1 << level);

  std::vector<BlockWeight> max_block_weights(current_k);
  if (current_k == input_p_ctx.k) {
    for (BlockID b = 0; b < current_k; ++b) {
      max_block_weights[b] = input_p_ctx.max_block_weight(b);
    }
  } else {
    BlockID cur_begin = 0;
    for (BlockID b = 0; b < current_k; ++b) {
      const BlockID num_sub_blocks = [&] {
        if (b < 2 * expanded_blocks) {
          const BlockID next_k =
              std::min<BlockID>(math::ceil2(current_k), next_block_weights.size());
          return partitioning::compute_final_k(b, next_k, next_block_weights.size());
        } else {
          return partitioning::compute_final_k(
              b - expanded_blocks, math::floor2(current_k), next_block_weights.size()
          );
        }
      }();

      const BlockID cur_end = cur_begin + num_sub_blocks;
      max_block_weights[b] = std::accumulate(
          next_block_weights.begin() + cur_begin, next_block_weights.begin() + cur_end, 0
      );

      DBG << "block " << b << ": aggregate weight for " << num_sub_blocks << " of "
          << next_block_weights.size() << " blocks when partitioning for the " << current_k
          << " v-cycle = " << max_block_weights[b];

      cur_begin = cur_end;
    }
  }
  return max_block_weights;
}

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

  std::vector<std::vector<BlockWeight>> vcycle_block_weights;

  for (auto it = steps.rbegin(); it != steps.rend(); ++it) {
    const BlockID current_k = *it;
    if (vcycle_block_weights.empty()) {
      vcycle_block_weights.push_back(compute_max_block_weights(current_k, _input_ctx.partition, {})
      );
    } else {
      auto &prev_block_weights = vcycle_block_weights.back();
      vcycle_block_weights.push_back(
          compute_max_block_weights(current_k, _input_ctx.partition, prev_block_weights)
      );
    }
  }

  std::reverse(vcycle_block_weights.begin(), vcycle_block_weights.end());

  std::size_t i = 0;
  for (const BlockID _ : steps) {
    {
      ctx.partition.set_epsilon(-1.0);
      ctx.partition.setup(_input_graph, std::move(vcycle_block_weights[i++]));
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

    DBG << "Block weights: ";
    for (const BlockID b : p_graph.blocks()) {
      DBG << "w(" << b << "): " << p_graph.block_weight(b)
          << "; max: " << ctx.partition.max_block_weight(b);
    }

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
