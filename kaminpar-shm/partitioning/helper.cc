/*******************************************************************************
 * Utility functions for common operations used by partitioning schemes.
 *
 * @file:   helper.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/partitioning/helper.h"

#include "kaminpar-shm/partition_utils.h"

#include "kaminpar-common/math.h"

namespace kaminpar::shm::partitioning::helper {
namespace {
SET_DEBUG(false);
SET_STATISTICS_FROM_GLOBAL();
} // namespace

void update_partition_context(PartitionContext &current_p_ctx, const PartitionedGraph &p_graph) {
  current_p_ctx.setup(p_graph.graph());
  current_p_ctx.k = p_graph.k();
  current_p_ctx.block_weights.setup(current_p_ctx, p_graph.final_ks());
}

PartitionedGraph
uncoarsen_once(Coarsener *coarsener, PartitionedGraph p_graph, PartitionContext &current_p_ctx) {
  SCOPED_TIMER("Uncoarsening");

  if (!coarsener->empty()) {
    p_graph = coarsener->uncoarsen(std::move(p_graph));
    update_partition_context(current_p_ctx, p_graph);
  }

  return p_graph;
}

void refine(Refiner *refiner, PartitionedGraph &p_graph, const PartitionContext &current_p_ctx) {
  SCOPED_TIMER("Refinement");
  refiner->initialize(p_graph);
  refiner->refine(p_graph, current_p_ctx);
}

PartitionedGraph bipartition(
    const Graph *graph,
    const BlockID final_k,
    const Context &input_ctx,
    GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool
) {
  InitialPartitioner partitioner(*graph, input_ctx, final_k, ip_m_ctx_pool.local().get());
  PartitionedGraph p_graph = partitioner.partition();
  ip_m_ctx_pool.local().put(partitioner.free());
  return p_graph;
}

void extend_partition_recursive(
    const Graph &graph,
    BlockArray &partition,
    const BlockID b0,
    const BlockID k,
    const BlockID final_k,
    const Context &input_ctx,
    graph::SubgraphMemory &subgraph_memory,
    const graph::SubgraphMemoryStartPosition position,
    TemporaryGraphExtractionBufferPool &extraction_pool,
    GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool
) {
  KASSERT(k > 1u);

  // obtain bipartition of current graph
  PartitionedGraph p_graph = bipartition(&graph, final_k, input_ctx, ip_m_ctx_pool);

  const BlockID final_k1 = p_graph.final_k(0);
  const BlockID final_k2 = p_graph.final_k(1);
  KASSERT(final_k1 > 0u);
  KASSERT(final_k2 > 0u);
  KASSERT(final_k == final_k1 + final_k2);

  std::array<BlockID, 2> ks{0, 0};
  std::tie(ks[0], ks[1]) = math::split_integral(k);
  KASSERT(ks[0] + ks[1] == k);
  KASSERT(ks[0] >= 1u);
  KASSERT(ks[1] >= 1u);
  KASSERT(final_k1 >= ks[0]);
  KASSERT(final_k2 >= ks[1]);

  // copy p_graph to partition -> replace b0 with b0 or b1
  std::array<BlockID, 2> b{b0, b0 + ks[0]};
  KASSERT(b[0] < input_ctx.partition.k);
  KASSERT(b[1] < input_ctx.partition.k);
  NodeID current_node = 0;
  for (std::size_t i = 0; i < partition.size(); ++i) {
    if (partition[i] == b0) {
      partition[i] = b[p_graph.block(current_node++)];
    }
  }
  KASSERT(current_node == p_graph.n());

  if (k > 2) {
    auto extraction =
        extract_subgraphs_sequential(p_graph, position, subgraph_memory, extraction_pool.local());
    const auto &subgraphs = extraction.subgraphs;
    const auto &positions = extraction.positions;

    for (const std::size_t i : {0, 1}) {
      if (ks[i] > 1) {
        extend_partition_recursive(
            subgraphs[i],
            partition,
            b[i],
            ks[i],
            p_graph.final_k(i),
            input_ctx,
            subgraph_memory,
            positions[i],
            extraction_pool,
            ip_m_ctx_pool
        );
      }
    }
  }
}

void extend_partition(
    PartitionedGraph &p_graph,
    const BlockID k_prime,
    const Context &input_ctx,
    PartitionContext &current_p_ctx,
    graph::SubgraphMemory &subgraph_memory,
    TemporaryGraphExtractionBufferPool &extraction_pool,
    GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool
) {
  SCOPED_TIMER("Initial partitioning");

  auto extraction = TIMED_SCOPE("Extract subgraphs") {
    return extract_subgraphs(p_graph, subgraph_memory);
  };
  const auto &subgraphs = extraction.subgraphs;
  const auto &mapping = extraction.node_mapping;
  const auto &positions = extraction.positions;

  START_TIMER("Allocation");
  scalable_vector<BlockArray> subgraph_partitions;
  for (const auto &subgraph : subgraphs) {
    subgraph_partitions.emplace_back(subgraph.n());
  }
  STOP_TIMER();

  START_TIMER("Bipartitioning");
  tbb::parallel_for(
      static_cast<BlockID>(0),
      static_cast<BlockID>(subgraphs.size()),
      [&](const BlockID b) {
        const auto &subgraph = subgraphs[b];
        const BlockID subgraph_k =
            (k_prime == input_ctx.partition.k) ? p_graph.final_k(b) : k_prime / p_graph.k();
        if (subgraph_k > 1) {
          KASSERT(subgraph_k <= p_graph.final_k(b));
          extend_partition_recursive(
              subgraph,
              subgraph_partitions[b],
              0,
              subgraph_k,
              p_graph.final_k(b),
              input_ctx,
              subgraph_memory,
              positions[b],
              extraction_pool,
              ip_m_ctx_pool
          );
        }
      }
  );
  STOP_TIMER();

  TIMED_SCOPE("Copy subgraph partitions") {
    graph::copy_subgraph_partitions(
        p_graph, subgraph_partitions, k_prime, input_ctx.partition.k, mapping
    );
  };
  update_partition_context(current_p_ctx, p_graph);

  KASSERT(p_graph.k() == k_prime);
}

// extend_partition with local memory allocation for subgraphs
void extend_partition(
    PartitionedGraph &p_graph,
    const BlockID k_prime,
    const Context &input_ctx,
    PartitionContext &current_p_ctx,
    TemporaryGraphExtractionBufferPool &extraction_pool,
    GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool
) {
  START_TIMER("Allocation");
  graph::SubgraphMemory memory{
      p_graph.n(),
      input_ctx.partition.k,
      p_graph.m(),
      p_graph.graph().is_node_weighted(),
      p_graph.graph().is_edge_weighted()};
  STOP_TIMER();
  extend_partition(
      p_graph, k_prime, input_ctx, current_p_ctx, memory, extraction_pool, ip_m_ctx_pool
  );
}

bool coarsen_once(
    Coarsener *coarsener,
    const Graph *graph,
    const Context &input_ctx,
    PartitionContext &current_p_ctx
) {
  SCOPED_TIMER("Coarsening");

  const NodeWeight max_cluster_weight =
      compute_max_cluster_weight(input_ctx.coarsening, *graph, input_ctx.partition);
  const auto [c_graph, shrunk] = coarsener->compute_coarse_graph(max_cluster_weight, 0);

  if (shrunk) {
    current_p_ctx.setup(*c_graph);
  }

  return shrunk;
}

BlockID compute_k_for_n(const NodeID n, const Context &input_ctx) {
  if (n < 2 * input_ctx.coarsening.contraction_limit) {
    return 2;
  } // catch special case where log is negative
  const BlockID k_prime = 1 << math::ceil_log2(n / input_ctx.coarsening.contraction_limit);
  return std::clamp(k_prime, static_cast<BlockID>(2), input_ctx.partition.k);
}

std::size_t compute_num_copies(
    const Context &input_ctx, const NodeID n, const bool converged, const std::size_t num_threads
) {
  KASSERT(num_threads > 0u);

  // sequential base case?
  const NodeID C = input_ctx.coarsening.contraction_limit;
  if (converged || n <= 2 * C) {
    return num_threads;
  }

  // parallel case
  const std::size_t f = 1 << static_cast<std::size_t>(std::ceil(std::log2(1.0 * n / C)));

  // continue with coarsening if the graph is still too large
  if (f > num_threads) {
    return 1;
  }

  // split into groups
  return num_threads / f;
}

std::size_t
select_best(const scalable_vector<PartitionedGraph> &p_graphs, const PartitionContext &p_ctx) {
  return select_best(p_graphs.begin(), p_graphs.end(), p_ctx);
}

std::size_t compute_num_threads_for_parallel_ip(const Context &input_ctx) {
  return math::floor2(static_cast<unsigned int>(
      1.0 * input_ctx.parallel.num_threads * input_ctx.partitioning.deep_initial_partitioning_load
  ));
}
} // namespace kaminpar::shm::partitioning::helper
