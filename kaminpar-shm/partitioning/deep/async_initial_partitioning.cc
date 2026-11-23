/*******************************************************************************
 * Implementation of the n / 2C < P phase of deep multilevel graph partitioning
 * scheduling the PE groups asynchronously.
 *
 * @file:   async_initial_partitioning.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/partitioning/deep/async_initial_partitioning.h"

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_invoke.h>
#include <tbb/task_arena.h>
#include <tbb/task_group.h>
#include <tbb/task_scheduler_observer.h>

#include "kaminpar-shm/coarsening/coarsener.h"
#include "kaminpar-shm/factories.h"
#include "kaminpar-shm/initial_partitioning/initial_bipartitioner_worker_pool.h"
#include "kaminpar-shm/partitioning/partition_utils.h"

namespace kaminpar::shm::partitioning {

namespace {

SET_DEBUG(true);

}

AsyncInitialPartitioner::AsyncInitialPartitioner(
    const Context &input_ctx,
    InitialBipartitionerWorkerPool &bipartitioner_pool,
    TemporarySubgraphMemoryEts &tmp_extraction_mem_pool_ets
)
    : _input_ctx(input_ctx),
      _refiner_ctx(input_ctx),
      _bipartitioner_pool(bipartitioner_pool),
      _tmp_extraction_mem_pool_ets(tmp_extraction_mem_pool_ets) {}

PartitionedGraph
AsyncInitialPartitioner::partition(const Coarsener *coarsener, const PartitionContext &p_ctx) {
  const std::size_t num_threads = compute_num_threads_for_parallel_ip(_input_ctx);
  _refiner_ctx = _input_ctx;
  _refiner_ctx.partition.n = coarsener->current().n();
  _refiner_ctx.partition.m = coarsener->current().m();
  _refiner_ctx.parallel.num_threads = static_cast<int>(num_threads);
  return split_and_join(coarsener, p_ctx, false, num_threads);
}

std::unique_ptr<Refiner> AsyncInitialPartitioner::get_refiner() {
  auto &pool = _refiner_pool_ets.local();
  if (!pool.empty()) {
    auto refiner = std::move(pool.back());
    pool.pop_back();
    return refiner;
  }
  return factory::create_refiner(_refiner_ctx);
}

void AsyncInitialPartitioner::put_refiner(std::unique_ptr<Refiner> refiner) {
  _refiner_pool_ets.local().push_back(std::move(refiner));
}

PartitionedGraph AsyncInitialPartitioner::partition_recursive(
    const Coarsener *parent_coarsener, PartitionContext &p_ctx, const std::size_t num_threads
) {
  const Graph *graph = &parent_coarsener->current();

  // Base case: only one thread left <=> compute bipartition
  if (num_threads == 1) {
    return _bipartitioner_pool.bipartition(graph, 0, 1, true);
  }

  // Otherwise, coarsen further and proceed recursively
  Context small_ctx = _input_ctx;
  small_ctx.partition.n = graph->n();
  small_ctx.partition.m = graph->m();
  small_ctx.parallel.num_threads = static_cast<int>(num_threads);

  auto coarsener = factory::create_coarsener(small_ctx, _input_ctx.partition);
  coarsener->initialize(graph);

  const bool shrunk = coarsener->coarsen();
  PartitionedGraph p_graph = split_and_join(coarsener.get(), p_ctx, !shrunk, num_threads);
  p_graph = coarsener->uncoarsen(std::move(p_graph));

  p_ctx = create_kway_context(_input_ctx, p_graph);
  auto refiner = get_refiner();
  refiner->initialize(p_graph);
  refiner->refine(p_graph, p_ctx);
  put_refiner(std::move(refiner));

  const BlockID k_prime = std::min(
      _input_ctx.partition.k,
      std::max<BlockID>(compute_k_for_n(p_graph.graph().n(), _input_ctx), num_threads)
  );

  if (p_graph.k() < k_prime) {
    if (_input_ctx.partitioning.use_lazy_subgraph_memory) {
      extend_partition_lazy_extraction(
          p_graph,
          k_prime,
          _input_ctx,
          _extraction_mem_pool_ets,
          _tmp_extraction_mem_pool_ets,
          _bipartitioner_pool,
          num_threads
      );
    } else {
      extend_partition(
          p_graph, k_prime, _input_ctx, _tmp_extraction_mem_pool_ets, _bipartitioner_pool, num_threads
      );
    }
    p_ctx = create_kway_context(_input_ctx, p_graph);
  }

  return p_graph;
}

PartitionedGraph AsyncInitialPartitioner::split_and_join(
    const Coarsener *coarsener,
    const PartitionContext &p_ctx,
    const bool converged,
    const std::size_t num_threads
) {
  const Graph *graph = &coarsener->current();
  const std::size_t num_copies = compute_num_copies(_input_ctx, graph->n(), converged, num_threads);
  const std::size_t threads_per_copy = num_threads / num_copies;

  // parallel recursion
  tbb::task_group tg;
  ScalableVector<PartitionedGraph> p_graphs(num_copies);
  ScalableVector<PartitionContext> p_ctx_copies(num_copies, p_ctx);

  for (std::size_t copy = 0; copy < num_copies; ++copy) {
    tg.run([this,
            copy, // `copy` must be captured by value
            coarsener,
            threads_per_copy,
            &p_graphs,
            &p_ctx_copies] {
      p_graphs[copy] = partition_recursive(coarsener, p_ctx_copies[copy], threads_per_copy);
    });
  }
  tg.wait();

  // Select best result
  const std::size_t best = select_best(p_graphs, p_ctx);
  return std::move(p_graphs[best]);
}

} // namespace kaminpar::shm::partitioning
