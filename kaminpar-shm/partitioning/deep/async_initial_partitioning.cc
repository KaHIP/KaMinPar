/*******************************************************************************
 * Implementation of the n / 2C < P phase of deep multilevel graph partitioning
 * scheduling the PE groups asynchronously.
 *
 * @file:   async_initial_partitioning.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/partitioning/deep/async_initial_partitioning.h"

namespace kaminpar::shm::partitioning {
AsyncInitialPartitioner::AsyncInitialPartitioner(
    const Context &input_ctx,
    GlobalInitialPartitionerMemoryPool &ip_m_ctx_pool,
    TemporaryGraphExtractionBufferPool &ip_extraction_pool
)
    : _input_ctx(input_ctx),
      _ip_m_ctx_pool(ip_m_ctx_pool),
      _ip_extraction_pool(ip_extraction_pool) {}

PartitionedGraph
AsyncInitialPartitioner::partition(const Coarsener *coarsener, const PartitionContext &p_ctx) {
  const std::size_t num_threads = helper::compute_num_threads_for_parallel_ip(_input_ctx);
  return split_and_join(coarsener, p_ctx, false, num_threads);
}

PartitionedGraph AsyncInitialPartitioner::partition_recursive(
    const Coarsener *parent_coarsener, PartitionContext &p_ctx, const std::size_t num_threads
) {
  const Graph *graph = parent_coarsener->coarsest_graph();

  // Base case: only one thread left <=> compute bipartition
  if (num_threads == 1) {
    return helper::bipartition(graph, _input_ctx.partition.k, _input_ctx, _ip_m_ctx_pool);
  }

  // Otherwise, coarsen further and proceed recursively
  auto coarsener = factory::create_coarsener(*graph, _input_ctx.coarsening);
  const bool shrunk = helper::coarsen_once(coarsener.get(), graph, _input_ctx, p_ctx);
  PartitionedGraph p_graph = split_and_join(coarsener.get(), p_ctx, !shrunk, num_threads);
  p_graph = helper::uncoarsen_once(coarsener.get(), std::move(p_graph), p_ctx, _input_ctx.partition);

  // The Context object is used to pre-allocate memory for the finest graph of the input hierarchy
  // Since this refiner is never used for the finest graph, we need to adjust the context to
  // prevent overallocation
  Context small_ctx = _input_ctx;
  small_ctx.partition.n = p_graph.n();
  small_ctx.partition.m = p_graph.m();
  auto refiner = factory::create_refiner(small_ctx);
  helper::refine(refiner.get(), p_graph, p_ctx);

  const BlockID k_prime = helper::compute_k_for_n(p_graph.n(), _input_ctx);
  if (p_graph.k() < k_prime) {
    helper::extend_partition(
        p_graph, k_prime, _input_ctx, p_ctx, _ip_extraction_pool, _ip_m_ctx_pool
    );
  }

  return p_graph;
}

PartitionedGraph AsyncInitialPartitioner::split_and_join(
    const Coarsener *coarsener,
    const PartitionContext &p_ctx,
    const bool converged,
    const std::size_t num_threads
) {
  const Graph *graph = coarsener->coarsest_graph();
  const std::size_t num_copies =
      helper::compute_num_copies(_input_ctx, graph->n(), converged, num_threads);
  const std::size_t threads_per_copy = num_threads / num_copies;

  // parallel recursion
  tbb::task_group tg;
  scalable_vector<PartitionedGraph> p_graphs(num_copies);
  scalable_vector<PartitionContext> p_ctx_copies(num_copies, p_ctx);

  for (std::size_t copy = 0; copy < num_copies; ++copy) {
    tg.run([this,
            copy,
            coarsener,
            threads_per_copy,
            &p_graphs,
            &p_ctx_copies] { // must capture copy by value!
      p_graphs[copy] = partition_recursive(coarsener, p_ctx_copies[copy], threads_per_copy);
    });
  }
  tg.wait();

  // select best result
  const std::size_t best = helper::select_best(p_graphs, p_ctx);
  return std::move(p_graphs[best]);
}
} // namespace kaminpar::shm::partitioning
