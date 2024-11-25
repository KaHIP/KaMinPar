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

SET_DEBUG(true);
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

    max_block_weights[coarse_block] = input_ctx.partition.total_max_block_weights(begin, end);
  }

  const bool is_toplevel_ctx = (p_graph.n() == input_ctx.partition.n);
  const bool relax_max_block_weights = !is_toplevel_ctx;

  PartitionContext new_p_ctx;
  new_p_ctx.setup(p_graph.graph(), std::move(max_block_weights), relax_max_block_weights);
  return new_p_ctx;
}

void extend_partition_recursive(
    const Graph &graph,
    StaticArray<BlockID> &partition,
    const BlockID current_block,
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
      bipartitioner_pool.bipartition(&graph, current_block, current_k, false);

  std::array<BlockID, 2> ks{0, 0};
  std::tie(ks[0], ks[1]) = math::split_integral(num_subblocks);
  std::array<BlockID, 2> b{current_block, current_block + ks[0]};

  { // Copy p_graph to partition
    NodeID node = 0;
    for (BlockID &block : partition) {
      block = (block == current_block) ? b[p_graph.block(node++)] : block;
    }
  }

  const BlockID final_k = compute_final_k(current_block, current_k, input_ctx.partition.k);
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
          b[i],
          ks[i],
          final_ks[i],
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
    const BlockID k_prime,     // extend to this many blocks
    const Context &input_ctx,  // stores input k
    PartitionContext &current_p_ctx,
    SubgraphMemoryEts &extraction_mem_pool_ets,
    TemporarySubgraphMemoryEts &tmp_extraction_mem_pool_ets,
    InitialBipartitionerWorkerPool &bipartitioner_pool,
    std::size_t num_active_threads
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
    while (k_prime > factor * p_graph.k() && num_active_threads > p_graph.k()) {
      extend_partition_lazy_extraction(
          p_graph,
          factor * p_graph.k(),
          input_ctx,
          current_p_ctx,
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
      const BlockID subgraph_k = (k_prime == input_ctx.partition.k) ? final_kb : k_prime / k;
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
          subgraph_k,
          final_kb,
          input_ctx,
          {.nodes_start_pos = 0, .edges_start_pos = 0},
          subgraph_memory,
          tmp_extraction_mem_pool_ets.local(),
          bipartitioner_pool,
          &timing
      );
    });

    if constexpr (kDebug) {
      const auto timings = dbg_timings_ets.combine([](auto &a, const auto &b) { return a += b; });
      const auto to_ms = [](const auto ns) {
        return static_cast<std::uint64_t>(ns / 1e6);
      };

      DBG << "bipartitioner_init_ms: " << to_ms(timings.bipartitioner_init_ms);
      DBG << "bipartitioner_ms:      " << to_ms(timings.bipartitioner_ms);
      DBG << "  total_ms:            " << to_ms(timings.ip_timings.total_ms);
      DBG << "  misc_ms:             " << to_ms(timings.ip_timings.misc_ms);
      DBG << "  coarsening_ms:       " << to_ms(timings.ip_timings.coarsening_ms);
      DBG << "    misc_ms:           " << to_ms(timings.ip_timings.coarsening_misc_ms);
      DBG << "    call_ms:           " << to_ms(timings.ip_timings.coarsening_call_ms);
      DBG << "      alloc_ms:        " << to_ms(timings.ip_timings.coarsening.alloc_ms);
      DBG << "      contract_ms:     " << to_ms(timings.ip_timings.coarsening.contract_ms);
      DBG << "      lp_ms:           " << to_ms(timings.ip_timings.coarsening.lp_ms);
      DBG << "      interleaved1:    " << to_ms(timings.ip_timings.coarsening.interleaved1_ms);
      DBG << "      interleaved2:    " << to_ms(timings.ip_timings.coarsening.interleaved2_ms);
      DBG << "  bipartitioning_ms:   " << to_ms(timings.ip_timings.bipartitioning_ms);
      DBG << "  uncoarsening_ms:     " << to_ms(timings.ip_timings.uncoarsening_ms);
      DBG << "graph_init_ms:         " << to_ms(timings.graph_init_ms);
      DBG << "extract_ms:            " << to_ms(timings.extract_ms);
      DBG << "copy_ms:               " << to_ms(timings.copy_ms);
      DBG << "misc_ms:               " << to_ms(timings.misc_ms);
    }
  };

  TIMED_SCOPE("Copy subgraph partitions") {
    SCOPED_HEAP_PROFILER("Copy subgraph partitions");
    p_graph = graph::copy_subgraph_partitions(
        std::move(p_graph), subgraph_partitions, k_prime, input_ctx.partition.k, mapping
    );
  };

  KASSERT(p_graph.k() == k_prime);
}

void extend_partition(
    PartitionedGraph &p_graph, // stores current k
    const BlockID k_prime,     // extend to this many blocks
    const Context &input_ctx,  // stores input k
    PartitionContext &current_p_ctx,
    graph::SubgraphMemory &subgraph_memory,
    TemporarySubgraphMemoryEts &tmp_extraction_mem_pool_ets,
    InitialBipartitionerWorkerPool &bipartitioner_pool,
    std::size_t num_active_threads
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
    while (k_prime > factor * p_graph.k() && num_active_threads > p_graph.k()) {
      extend_partition(
          p_graph,
          factor * p_graph.k(),
          input_ctx,
          current_p_ctx,
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
          (k_prime == input_ctx.partition.k) ? final_kb : k_prime / p_graph.k();

      if (subgraph_k > 1) {
        DBG << "initial extend_partition_recursive() for block " << b << ", final k " << final_kb
            << ", subgraph k " << subgraph_k << ", weight " << p_graph.block_weight(b) << " /// "
            << subgraph.total_node_weight();

        extend_partition_recursive(
            subgraph,
            subgraph_partitions[b],
            0,
            subgraph_k,
            final_kb,
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
        std::move(p_graph), subgraph_partitions, k_prime, input_ctx.partition.k, mapping
    );
  };

  if constexpr (kDebug) {
    const auto timings = timings_ets.combine([](auto &a, const auto &b) { return a += b; });
    const auto to_ms = [](const auto ns) {
      return static_cast<std::uint64_t>(ns / 1e6);
    };

    LOG << "bipartitioner_init_ms: " << to_ms(timings.bipartitioner_init_ms);
    LOG << "bipartitioner_ms:      " << to_ms(timings.bipartitioner_ms);
    LOG << "  total_ms:            " << to_ms(timings.ip_timings.total_ms);
    LOG << "  misc_ms:             " << to_ms(timings.ip_timings.misc_ms);
    LOG << "  coarsening_ms:       " << to_ms(timings.ip_timings.coarsening_ms);
    LOG << "    misc_ms:           " << to_ms(timings.ip_timings.coarsening_misc_ms);
    LOG << "    call_ms:           " << to_ms(timings.ip_timings.coarsening_call_ms);
    LOG << "      alloc_ms:        " << to_ms(timings.ip_timings.coarsening.alloc_ms);
    LOG << "      contract_ms:     " << to_ms(timings.ip_timings.coarsening.contract_ms);
    LOG << "      lp_ms:           " << to_ms(timings.ip_timings.coarsening.lp_ms);
    LOG << "      interleaved1:    " << to_ms(timings.ip_timings.coarsening.interleaved1_ms);
    LOG << "      interleaved2:    " << to_ms(timings.ip_timings.coarsening.interleaved2_ms);
    LOG << "  bipartitioning_ms:   " << to_ms(timings.ip_timings.bipartitioning_ms);
    LOG << "  uncoarsening_ms:     " << to_ms(timings.ip_timings.uncoarsening_ms);
    LOG << "graph_init_ms:         " << to_ms(timings.graph_init_ms);
    LOG << "extract_ms:            " << to_ms(timings.extract_ms);
    LOG << "copy_ms:               " << to_ms(timings.copy_ms);
    LOG << "misc_ms:               " << to_ms(timings.misc_ms);
  }

  KASSERT(p_graph.k() == k_prime);
}

// extend_partition with local memory allocation for subgraphs
void extend_partition(
    PartitionedGraph &p_graph,
    const BlockID k_prime,
    const Context &input_ctx,
    PartitionContext &current_p_ctx,
    TemporarySubgraphMemoryEts &tmp_extraction_mem_pool_ets,
    InitialBipartitionerWorkerPool &bipartitioner_pool,
    std::size_t num_active_threads
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
      k_prime,
      input_ctx,
      current_p_ctx,
      memory,
      tmp_extraction_mem_pool_ets,
      bipartitioner_pool,
      num_active_threads
  );
}

std::size_t
select_best(const ScalableVector<PartitionedGraph> &p_graphs, const PartitionContext &p_ctx) {
  return select_best(p_graphs.begin(), p_graphs.end(), p_ctx);
}

} // namespace kaminpar::shm::partitioning
