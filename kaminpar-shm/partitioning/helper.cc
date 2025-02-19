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
#include "kaminpar-common/parallel/algorithm.h"
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

PartitionContext create_twoway_context(
    const Context &input_ctx,
    const BlockID current_block,
    const BlockID current_k,
    const AbstractGraph &graph
) {
  // Through recursive bipartitioning, `current_block` (i.e., `graph`) will be subdivided further
  // into a range of sub-blocks: R = [first_sub_block, first_invalid_sub_block).
  const BlockID first_sub_block =
      partitioning::compute_first_sub_block(current_block, current_k, input_ctx.partition.k);
  const BlockID first_invalid_sub_block = partitioning::compute_first_invalid_sub_block(
      current_block, current_k, input_ctx.partition.k
  );
  const BlockID num_sub_blocks =
      partitioning::compute_final_k(current_block, current_k, input_ctx.partition.k);

  // The first `num_sub_blocks_b0` of `R` will be descendands of the first block of the bipartition
  // that we are about to compute; the remaining ones will be descendands of the second block.
  const auto [num_sub_blocks_b0, num_sub_blocks_b1] = math::split_integral(num_sub_blocks);

  // Based on this information, we can compute the maximum block weights by summing all maximum
  // block weights of the corresponding sub-blocks.
  std::vector<BlockWeight> max_block_weights{
      input_ctx.partition.total_max_block_weights(
          first_sub_block, first_sub_block + num_sub_blocks_b0
      ),
      input_ctx.partition.total_max_block_weights(
          first_sub_block + num_sub_blocks_b0, first_invalid_sub_block
      )
  };

  DBG << "[" << current_block << "/" << current_k << "] Current weight "
      << graph.total_node_weight() << ", spans sub-blocks [" << first_sub_block << ", "
      << first_invalid_sub_block << "), split max weight "
      << input_ctx.partition.total_max_block_weights(first_sub_block, first_invalid_sub_block)
      << " into " << max_block_weights[0] << " and " << max_block_weights[1];

  PartitionContext p_ctx;

  // @todo: how to adapt the inferred epsilon when dealing with arbitrary block weights?
  if (input_ctx.partition.has_uniform_block_weights() &&
      input_ctx.initial_partitioning.use_adaptive_epsilon) {
    // It can be beneficial to artifically "restrict" the maximum block weights of *this*
    // bipartition, ensuring that there is enough wiggle room for further bipartitioning of the
    // sub-blocks: this is based on the "adapted epsilon" strategy of KaHyPar.
    const double base = (1.0 + input_ctx.partition.inferred_epsilon()) * num_sub_blocks *
                        input_ctx.partition.total_node_weight / input_ctx.partition.k /
                        graph.total_node_weight();
    const double exponent = 1.0 / math::ceil_log2(num_sub_blocks);
    const double epsilon_prime = std::pow(base, exponent) - 1.0;
    const double adapted_eps = std::max(epsilon_prime, 0.0001);

    const BlockWeight total_max_weight = max_block_weights[0] + max_block_weights[1];
    std::array<double, 2> max_weight_ratios = {
        1.0 * max_block_weights[0] / total_max_weight, 1.0 * max_block_weights[1] / total_max_weight
    };

    for (const BlockID b : {0, 1}) {
      max_block_weights[b] = (1.0 + adapted_eps) * graph.total_node_weight() * max_weight_ratios[b];
    }

    DBG << "[" << current_block << "/" << current_k << "]-> adapted epsilon from "
        << input_ctx.partition.epsilon() << " to " << adapted_eps
        << ", changing max block weights to " << max_block_weights[0] << " + "
        << max_block_weights[1] << ", will be relaxed with parameters max node weight "
        << graph.max_node_weight();

    p_ctx.setup(graph, std::move(max_block_weights), true);
  } else {
    DBG << "[" << current_block << "/" << current_k
        << "]-> using original epsilon: " << input_ctx.partition.epsilon()
        << ", inferred from max block weights " << max_block_weights[0] << " and "
        << max_block_weights[1];

    p_ctx.setup(graph, std::move(max_block_weights), true);
  }

  return p_ctx;
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
    BipartitionTimingInfo *timings 
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
          << ", subgraph k " << subgraph_k << ", weight " << p_graph.block_weight(b) << " of "
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

void complete_partial_extend_partition(
    PartitionedGraph &p_graph,
    const Context &input_ctx,
    SubgraphMemoryEts &extraction_mem_pool_ets,
    TemporarySubgraphMemoryEts &tmp_extraction_mem_pool_ets,
    InitialBipartitionerWorkerPool &bipartitioner_pool
) {
  SCOPED_TIMER("Initial partitioning");
  const BlockID current_k = p_graph.k();

  DBG << "Complete partial extend_partition() for k=" << current_k
      << " to k=" << input_ctx.partition.k;

  if (current_k == input_ctx.partition.k || math::is_power_of_2(current_k)) {
    return;
  }

  auto [mapping, block_nodes_offset, block_nodes, block_num_edges] = TIMED_SCOPE("Preprocessing") {
    SCOPED_HEAP_PROFILER("Preprocessing");
    return graph::lazy_extract_subgraphs_preprocessing(p_graph);
  };

  auto subgraph_partitions = TIMED_SCOPE("Allocation") {
    SCOPED_HEAP_PROFILER("Allocation");

    ScalableVector<StaticArray<BlockID>> subgraph_partitions;
    for (BlockID b = 0; b < current_k; ++b) {
      const NodeID num_block_nodes = block_nodes_offset[b + 1] - block_nodes_offset[b];
      subgraph_partitions.emplace_back(num_block_nodes);
    }

    return subgraph_partitions;
  };

  const int level = math::floor_log2(current_k);
  const BlockID expanded_blocks = current_k - (1 << level);
  const BlockID prev_current_k = math::floor2(current_k);
  const BlockID desired_k = std::min(math::ceil2(current_k), input_ctx.partition.k);

  TIMED_SCOPE("Bipartitioning") {
    SCOPED_HEAP_PROFILER("Bipartitioning");

    tbb::parallel_for<BlockID>(0, current_k, [&](const BlockID b) {
      // This block is already on the next level of the recursive bipartitioning tree
      if (b < 2 * expanded_blocks) {
        return;
      }

      const BlockID prev_b = b - expanded_blocks;

      const BlockID final_kb = compute_final_k(prev_b, prev_current_k, input_ctx.partition.k);
      const BlockID subgraph_k = (desired_k == input_ctx.partition.k) ? final_kb : 2;
      if (subgraph_k <= 1) {
        return;
      }

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

      DBG << "Initial extend_partition_recursive() for abs block " << b << " with final k "
          << final_kb << ", subgraph k " << subgraph_k << ", weight " << p_graph.block_weight(b)
          << " of " << subgraph.total_node_weight();

      extend_partition_recursive(
          subgraph,
          subgraph_partitions[b],
          0,
          prev_b,
          subgraph_k,
          prev_current_k,
          input_ctx,
          {.nodes_start_pos = 0, .edges_start_pos = 0},
          subgraph_memory,
          tmp_extraction_mem_pool_ets.local(),
          bipartitioner_pool,
          nullptr
      );
    });
  };

  TIMED_SCOPE("Copy subgraph partitions") {
    SCOPED_HEAP_PROFILER("Copy subgraph partitions");
    std::vector<BlockID> k0(p_graph.k() + 1, desired_k / p_graph.k());
    k0.front() = 0;

    for (const BlockID b : p_graph.blocks()) {
      if (b < 2 * expanded_blocks) {
        k0[b + 1] = 1;
      } else {
        k0[b + 1] = std::min<BlockID>(
            2, compute_final_k(b - expanded_blocks, prev_current_k, input_ctx.partition.k)
        );
      }
    }

    DBG << "Block offsets: " << k0;

    parallel::prefix_sum(k0.begin(), k0.end(), k0.begin());

    StaticArray<BlockID> partition = p_graph.take_raw_partition();
    p_graph.pfor_nodes([&](const NodeID u) {
      const BlockID b = partition[u];
      const NodeID s_u = mapping[u];
      partition[u] = k0[b] + subgraph_partitions[b][s_u];
    });

    p_graph = PartitionedGraph(p_graph.graph(), desired_k, std::move(partition));
  };
}

} // namespace kaminpar::shm::partitioning
