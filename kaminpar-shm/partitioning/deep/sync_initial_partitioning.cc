/*******************************************************************************
 * Implementation of the n / 2C < P phase of deep multilevel graph partitioning
 * scheduling the PE groups synchronously.
 *
 * @file:   sync_initial_partitioning.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 ******************************************************************************/
#include "kaminpar-shm/partitioning/deep/sync_initial_partitioning.h"

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

SET_DEBUG(false);

}

SyncInitialPartitioner::SyncInitialPartitioner(
    const Context &input_ctx,
    InitialBipartitionerWorkerPool &bipartitioner_pool,
    TemporarySubgraphMemoryEts &tmp_extraction_mem_pool_ets
)
    : _input_ctx(input_ctx),
      _bipartitioner_pool(bipartitioner_pool),
      _tmp_extraction_mem_pool_ets(tmp_extraction_mem_pool_ets) {}

PartitionedGraph
SyncInitialPartitioner::partition(const Coarsener *coarsener, const PartitionContext &p_ctx) {
  const std::size_t num_threads = compute_num_threads_for_parallel_ip(_input_ctx);

  std::vector<std::vector<std::unique_ptr<Coarsener>>> coarseners(1);
  std::vector<PartitionContext> current_p_ctxs;
  coarseners[0].push_back(duplicate_coarsener(coarsener));
  current_p_ctxs.push_back(p_ctx);

  std::size_t num_current_threads = num_threads;
  std::size_t num_current_copies = 1;
  std::atomic<bool> converged = false;

  std::vector<std::size_t> num_local_copies_record;

  while (num_current_copies < num_threads) {
    const NodeID n = coarseners.back()[0]->current().n();
    const std::size_t num_local_copies =
        compute_num_copies(_input_ctx, n, converged, num_current_threads);
    num_local_copies_record.push_back(num_local_copies);

    // Create coarseners and partition contexts for next coarsening iteration
    coarseners.emplace_back(num_current_copies * num_local_copies);
    auto &next_coarseners = coarseners.back();
    auto &current_coarseners = coarseners[coarseners.size() - 2];
    std::vector<PartitionContext> next_p_ctxs(num_current_copies * num_local_copies);

    tbb::parallel_for(static_cast<std::size_t>(0), num_current_copies, [&](const std::size_t i) {
      const std::size_t next_i = i * num_local_copies;
      for (std::size_t next_j = next_i; next_j < next_i + num_local_copies; ++next_j) {
        next_coarseners[next_j] = duplicate_coarsener(current_coarseners[i].get());
        next_p_ctxs[next_j] = current_p_ctxs[i];
      }
    });

    std::swap(current_p_ctxs, next_p_ctxs);

    num_current_threads /= num_local_copies;
    num_current_copies *= num_local_copies;

    // Perform coarsening iteration, converge if all coarseners converged
    converged = true;
    tbb::parallel_for(static_cast<std::size_t>(0), num_current_copies, [&](const std::size_t i) {
      const bool shrunk = next_coarseners[i]->coarsen();
      if (shrunk) {
        converged = false;
      }
    });
  }

  // Perform initial bipartition on every graph
  std::vector<PartitionedGraph> current_p_graphs(num_threads);
  tbb::parallel_for(static_cast<std::size_t>(0), num_threads, [&](const std::size_t i) {
    auto &current_coarseners = coarseners.back();
    const Graph *graph = &current_coarseners[i]->current();
    current_p_graphs[i] = _bipartitioner_pool.bipartition(graph, 0, 1, true);
  });

  // Uncoarsen and join graphs
  while (!num_local_copies_record.empty()) {
    const std::size_t num_local_copies = num_local_copies_record.back();
    num_local_copies_record.pop_back();

    auto &current_coarseners = coarseners.back();

    // Uncoarsen and refine
    tbb::parallel_for(static_cast<std::size_t>(0), num_current_copies, [&](const std::size_t i) {
      auto &p_graph = current_p_graphs[i];
      auto &coarsener = current_coarseners[i];
      auto &p_ctx = current_p_ctxs[i];
      p_graph = coarsener->uncoarsen(std::move(p_graph));
      p_ctx = create_kway_context(_input_ctx, p_graph);

      // The Context object is used to pre-allocate memory for the finest graph of the input
      // hierarchy Since this refiner is never used for the finest graph, we need to adjust the
      // context to prevent overallocation
      Context small_ctx = _input_ctx;
      small_ctx.partition.n = p_graph.n();
      small_ctx.partition.m = p_graph.m();

      auto refiner = factory::create_refiner(small_ctx);
      refiner->initialize(p_graph);
      refiner->refine(p_graph, p_ctx);

      // extend partition
      const BlockID k_prime = compute_k_for_n(p_graph.n(), _input_ctx);
      if (p_graph.k() < k_prime) {
        extend_partition(
            p_graph,
            k_prime,
            _input_ctx,
            _tmp_extraction_mem_pool_ets,
            _bipartitioner_pool,
            num_threads
        );
        p_ctx = create_kway_context(_input_ctx, p_graph);
      }
    });

    num_current_copies /= num_local_copies;
    num_current_threads *= num_local_copies;

    std::vector<PartitionContext> next_p_ctxs(num_current_copies);
    std::vector<PartitionedGraph> next_p_graphs(num_current_copies);

    tbb::parallel_for<std::size_t>(0, num_current_copies, [&](const std::size_t i) {
      // Join
      const std::size_t start_pos = i * num_local_copies;
      PartitionContext &p_ctx = current_p_ctxs[start_pos];
      const std::size_t pos = select_best(
          current_p_graphs.begin() + start_pos,
          current_p_graphs.begin() + start_pos + num_local_copies,
          p_ctx
      );
      PartitionedGraph &p_graph = current_p_graphs[start_pos + pos];

      // Store
      next_p_ctxs[i] = std::move(p_ctx);
      next_p_graphs[i] = std::move(p_graph);
    });

    std::swap(current_p_ctxs, next_p_ctxs);
    std::swap(current_p_graphs, next_p_graphs);
    coarseners.pop_back();
  }

  KASSERT(coarseners.size() == 1u, "", assert::light);
  KASSERT(&(current_p_graphs.front().graph()) == &coarsener->current(), "", assert::light);
  return std::move(current_p_graphs.front());
}

std::unique_ptr<Coarsener> SyncInitialPartitioner::duplicate_coarsener(const Coarsener *coarsener) {
  auto duplication = factory::create_coarsener(_input_ctx);
  duplication->initialize(&coarsener->current());
  return duplication;
}

} // namespace kaminpar::shm::partitioning
