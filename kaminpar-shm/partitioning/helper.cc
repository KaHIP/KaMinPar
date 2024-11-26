/*******************************************************************************
 * Utility functions for common operations used by partitioning schemes.
 *
 * @file:   helper.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/partitioning/helper.h"

#include "kaminpar-shm/partitioning/partition_utils.h"

#include "kaminpar-common/heap_profiler.h"
#include "kaminpar-common/math.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm::partitioning {

namespace {

SET_DEBUG(false);
SET_STATISTICS_FROM_GLOBAL();

} // namespace

PartitionContext create_kway_context(const Context &input_ctx, const PartitionedGraph &p_graph) {
  const BlockID input_k = input_ctx.partition.k;
  const BlockID current_k = p_graph.k();

  std::vector<BlockWeight> max_block_weights(p_graph.k());
  BlockID cur_fine_block = 0;
  for (const BlockID coarse_block : p_graph.blocks()) {
    const BlockID num = compute_final_k(coarse_block, current_k, input_k);
    const BlockID begin = cur_fine_block;
    const BlockID end = cur_fine_block + num;
    cur_fine_block += num;

    max_block_weights[coarse_block] =
        input_ctx.partition.total_unrelaxed_max_block_weights(begin, end);

    if (p_graph.k() != input_ctx.partition.k) { // @todo
      max_block_weights[coarse_block] += end - begin;
    }
  }

  const bool is_toplevel_ctx = (p_graph.n() == input_ctx.partition.n);
  const bool relax_max_block_weights = !is_toplevel_ctx;

  PartitionContext new_p_ctx;
  new_p_ctx.setup(p_graph.graph(), std::move(max_block_weights), relax_max_block_weights);

  // @todo
  if (input_ctx.partition.has_epsilon()) {
    new_p_ctx.set_epsilon(input_ctx.partition.epsilon());
  }

  return new_p_ctx;
}

void extend_partition_recursive(
    const Graph &graph,
    StaticArray<BlockID> &partition,
    const BlockID current_rel_block,
    const BlockID current_abs_block,
    const BlockID num_subblocks,
    const BlockID current_k,
    const Context &input_ctx,
    const graph::SubgraphMemoryStartPosition position,
    graph::SubgraphMemory &subgraph_memory,
    graph::TemporarySubgraphMemory &tmp_extraction_mem_pool,
    InitialBipartitionerWorkerPool &bipartitioner_pool,
    BipartitionTimingInfo *timings = nullptr
) {
  KASSERT(num_subblocks > 1u);

  PartitionedGraph p_graph =
      bipartitioner_pool.bipartition(&graph, current_abs_block, current_k, false);

  std::array<BlockID, 2> ks{0, 0};
  std::tie(ks[0], ks[1]) = math::split_integral(num_subblocks);
  std::array<BlockID, 2> rel_b{current_rel_block, current_rel_block + ks[0]};

  // @todo should be correct, but needs clean ups
  std::array<BlockID, 2> abs_b;
  if (2 * current_k >= input_ctx.partition.k) {
    abs_b = {
        compute_first_sub_block(current_abs_block, current_k, input_ctx.partition.k),
        compute_first_sub_block(current_abs_block, current_k, input_ctx.partition.k) + 1
    };
  } else {
    abs_b = {2 * current_abs_block, 2 * current_abs_block + 1};
  }

  DBG << "[k=" << current_k << "] Apply partition of block abs/" << current_abs_block << "-rel/"
      << current_rel_block << " into blocks abs/" << abs_b[0] << "-rel/" << rel_b[0] << " and abs/"
      << abs_b[1] << "-rel/" << rel_b[1] << ", num sub-blocks: " << num_subblocks;

  { // Copy p_graph to partition
    NodeID node = 0;
    for (BlockID &block : partition) {
      block = (block == current_rel_block) ? rel_b[p_graph.block(node++)] : block;
    }
  }

  const BlockID final_k = compute_final_k(current_abs_block, current_k, input_ctx.partition.k);
  std::array<BlockID, 2> final_ks{0, 0};
  std::tie(final_ks[0], final_ks[1]) = math::split_integral(final_k);

  if (num_subblocks > 2) {
    auto [subgraphs, positions] = extract_subgraphs_sequential(
        p_graph, final_ks, position, subgraph_memory, tmp_extraction_mem_pool
    );

    for (const std::size_t i : {0, 1}) {
      if (ks[i] <= 1) {
        continue;
      }

      extend_partition_recursive(
          subgraphs[i],
          partition,
          rel_b[i],
          abs_b[i],
          ks[i],
          partitioning::compute_next_k(current_k, input_ctx),
          input_ctx,
          positions[i],
          subgraph_memory,
          tmp_extraction_mem_pool,
          bipartitioner_pool,
          timings
      );
    }
  }
}

void extend_partition_lazy_extraction(
    PartitionedGraph &p_graph, // stores current k
    const BlockID desired_k,   // extend to this many blocks
    const Context &input_ctx,  // stores input k
    SubgraphMemoryEts &extraction_mem_pool_ets,
    TemporarySubgraphMemoryEts &tmp_extraction_mem_pool_ets,
    InitialBipartitionerWorkerPool &bipartitioner_pool,
    const int num_active_threads
) {
  if (input_ctx.partitioning.min_consecutive_seq_bipartitioning_levels > 0) {
    // Depending on the coarsening level and the deep multilevel implementation, it can occur that
    // this function is called with more threads than blocks in the graph partition. To avoid
    // wasting threads, we only extend the partition a little at first, and then recurse until all
    // threads can work on independent blocks.
    // "min_consecutive_seq_bipartitioning_levels" parameterizes the term "a little": when set to 1,
    // we have the most amount of parallelization, but waste time by re-extracting the block-induced
    // subgraphs from the partitioned graph; larger values do this less often at the cost of wasting
    // more parallel compute resources.
    // @todo change async_initial_partitioning.{cc, h} to make this obsolete ...
    const int factor = 2 << (input_ctx.partitioning.min_consecutive_seq_bipartitioning_levels - 1);
    while (desired_k > factor * p_graph.k() &&
           static_cast<BlockID>(num_active_threads) > p_graph.k()) {
      extend_partition_lazy_extraction(
          p_graph,
          factor * p_graph.k(),
          input_ctx,
          extraction_mem_pool_ets,
          tmp_extraction_mem_pool_ets,
          bipartitioner_pool,
          num_active_threads
      );
    }
  }

  SCOPED_TIMER("Initial partitioning");
  const BlockID k = p_graph.k();

  auto [mapping, block_nodes_offset, block_nodes, block_num_edges] = TIMED_SCOPE("Preprocessing") {
    SCOPED_HEAP_PROFILER("Preprocessing");
    return graph::lazy_extract_subgraphs_preprocessing(p_graph);
  };

  auto subgraph_partitions = TIMED_SCOPE("Allocation") {
    SCOPED_HEAP_PROFILER("Allocation");

    ScalableVector<StaticArray<BlockID>> subgraph_partitions;
    for (BlockID b = 0; b < k; ++b) {
      const NodeID num_block_nodes = block_nodes_offset[b + 1] - block_nodes_offset[b];
      subgraph_partitions.emplace_back(num_block_nodes);
    }

    return subgraph_partitions;
  };

  TIMED_SCOPE("Bipartitioning") {
    SCOPED_HEAP_PROFILER("Bipartitioning");
    tbb::enumerable_thread_specific<BipartitionTimingInfo> dbg_timings_ets;

    tbb::parallel_for<BlockID>(0, k, [&](const BlockID b) {
      const BlockID final_kb = compute_final_k(b, k, input_ctx.partition.k);
      const BlockID subgraph_k = (desired_k == input_ctx.partition.k) ? final_kb : desired_k / k;
      if (subgraph_k <= 1) {
        return;
      }

      auto &timing = dbg_timings_ets.local();
      auto &subgraph_memory = extraction_mem_pool_ets.local();

      const NodeID num_block_nodes = block_nodes_offset[b + 1] - block_nodes_offset[b];
      const NodeID subgraph_memory_n = num_block_nodes + final_kb;
      const EdgeID num_block_edges = block_num_edges[b];

      if (subgraph_memory.nodes.size() < subgraph_memory_n) {
        subgraph_memory.nodes.resize(subgraph_memory_n, static_array::seq, static_array::noinit);

        if (p_graph.is_node_weighted()) {
          subgraph_memory.node_weights.resize(
              subgraph_memory_n, static_array::seq, static_array::noinit
          );
        }
      }

      if (subgraph_memory.edges.size() < num_block_edges) {
        subgraph_memory.edges.resize(num_block_edges, static_array::seq, static_array::noinit);

        if (p_graph.is_edge_weighted()) {
          subgraph_memory.edge_weights.resize(
              num_block_edges, static_array::seq, static_array::noinit
          );
        }
      }

      const StaticArray<NodeID> local_block_nodes(
          num_block_nodes, block_nodes.data() + block_nodes_offset[b]
      );
      const auto subgraph =
          graph::extract_subgraph(p_graph, b, local_block_nodes, mapping, subgraph_memory);

      DBG << "initial extend_partition_recursive() for block " << b << ", final k " << final_kb
          << ", subgraph k " << subgraph_k << ", weight " << p_graph.block_weight(b) << " /// "
          << subgraph.total_node_weight();

      extend_partition_recursive(
          subgraph,
          subgraph_partitions[b],
          0,
          b,
          subgraph_k,
          p_graph.k(),
          input_ctx,
          {.nodes_start_pos = 0, .edges_start_pos = 0},
          subgraph_memory,
          tmp_extraction_mem_pool_ets.local(),
          bipartitioner_pool,
          &timing
      );
    });
  };

  TIMED_SCOPE("Copy subgraph partitions") {
    SCOPED_HEAP_PROFILER("Copy subgraph partitions");
    p_graph = graph::copy_subgraph_partitions(
        std::move(p_graph), subgraph_partitions, desired_k, input_ctx.partition.k, mapping
    );
  };

  KASSERT(p_graph.k() == desired_k);
}

void extend_partition(
    PartitionedGraph &p_graph, // stores current k
    const BlockID desired_k,   // extend to this many blocks
    const Context &input_ctx,  // stores input k
    graph::SubgraphMemory &subgraph_memory,
    TemporarySubgraphMemoryEts &tmp_extraction_mem_pool_ets,
    InitialBipartitionerWorkerPool &bipartitioner_pool,
    const int num_active_threads
) {
  if (input_ctx.partitioning.min_consecutive_seq_bipartitioning_levels > 0) {
    // Depending on the coarsening level and the deep multilevel implementation, it can occur that
    // this function is called with more threads than blocks in the graph partition. To avoid
    // wasting threads, we only extend the partition a little at first, and then recurse until all
    // threads can work on independent blocks.
    // "min_consecutive_seq_bipartitioning_levels" parameterizes the term "a little": when set to 1,
    // we have the most amount of parallelization, but waste time by re-extracting the block-induced
    // subgraphs from the partitioned graph; larger values do this less often at the cost of wasting
    // more parallel compute resources.
    // @todo change async_initial_partitioning.{cc, h} to make this obsolete ...
    const int factor = 2 << (input_ctx.partitioning.min_consecutive_seq_bipartitioning_levels - 1);
    while (desired_k > factor * p_graph.k() &&
           static_cast<BlockID>(num_active_threads) > p_graph.k()) {
      extend_partition(
          p_graph,
          factor * p_graph.k(),
          input_ctx,
          subgraph_memory,
          tmp_extraction_mem_pool_ets,
          bipartitioner_pool,
          num_active_threads
      );
    }
  }

  SCOPED_TIMER("Initial partitioning");

  const auto [subgraphs, mapping, positions] = TIMED_SCOPE("Extract subgraphs") {
    SCOPED_HEAP_PROFILER("Extract subgraphs");
    return extract_subgraphs(p_graph, input_ctx.partition.k, subgraph_memory);
  };

  auto subgraph_partitions = TIMED_SCOPE("Allocation") {
    SCOPED_HEAP_PROFILER("Allocation");

    ScalableVector<StaticArray<BlockID>> subgraph_partitions;
    for (const auto &subgraph : subgraphs) {
      subgraph_partitions.emplace_back(subgraph.n());
    }

    return subgraph_partitions;
  };

  tbb::enumerable_thread_specific<BipartitionTimingInfo> timings_ets;
  TIMED_SCOPE("Bipartitioning") {
    SCOPED_HEAP_PROFILER("Bipartitioning");

    tbb::parallel_for<BlockID>(0, subgraphs.size(), [&](const BlockID b) {
      BipartitionTimingInfo &timing = timings_ets.local();

      const auto &subgraph = subgraphs[b];
      const BlockID final_kb = compute_final_k(b, p_graph.k(), input_ctx.partition.k);

      const BlockID subgraph_k =
          (desired_k == input_ctx.partition.k) ? final_kb : desired_k / p_graph.k();

      if (subgraph_k > 1) {
        DBG << "initial extend_partition_recursive() for block " << b << ", final k " << final_kb
            << ", subgraph k " << subgraph_k << ", weight " << p_graph.block_weight(b) << " /// "
            << subgraph.total_node_weight();

        extend_partition_recursive(
            subgraph,
            subgraph_partitions[b],
            0,
            b,
            subgraph_k,
            p_graph.k(),
            input_ctx,
            positions[b],
            subgraph_memory,
            tmp_extraction_mem_pool_ets.local(),
            bipartitioner_pool,
            &timing
        );
      }
    });
  };

  TIMED_SCOPE("Copy subgraph partitions") {
    SCOPED_HEAP_PROFILER("Copy subgraph partitions");
    p_graph = graph::copy_subgraph_partitions(
        std::move(p_graph), subgraph_partitions, desired_k, input_ctx.partition.k, mapping
    );
  };

  KASSERT(p_graph.k() == desired_k);
}

// extend_partition with local memory allocation for subgraphs
void extend_partition(
    PartitionedGraph &p_graph,
    const BlockID desired_k,
    const Context &input_ctx,
    TemporarySubgraphMemoryEts &tmp_extraction_mem_pool_ets,
    InitialBipartitionerWorkerPool &bipartitioner_pool,
    const int num_active_threads
) {
  graph::SubgraphMemory memory;

  memory.resize(
      p_graph.n(),
      input_ctx.partition.k,
      p_graph.m(),
      p_graph.graph().is_node_weighted(),
      p_graph.graph().is_edge_weighted()
  );

  extend_partition(
      p_graph,
      desired_k,
      input_ctx,
      memory,
      tmp_extraction_mem_pool_ets,
      bipartitioner_pool,
      num_active_threads
  );
}

} // namespace kaminpar::shm::partitioning
