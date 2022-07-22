/*******************************************************************************
 * @file:   parallel_initial_partitioner.cc
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:
 ******************************************************************************/
#include "kaminpar/partitioning_scheme/parallel_initial_partitioner.h"

namespace kaminpar::shm::partitioning {
ParallelInitialPartitioner::ParallelInitialPartitioner(
    const Context& input_ctx, GlobalInitialPartitionerMemoryPool& ip_m_ctx_pool,
    TemporaryGraphExtractionBufferPool& ip_extraction_pool)
    : _input_ctx{input_ctx},
      _ip_m_ctx_pool{ip_m_ctx_pool},
      _ip_extraction_pool{ip_extraction_pool} {}

PartitionedGraph ParallelInitialPartitioner::partition(const ICoarsener* coarsener, const PartitionContext& p_ctx) {
    const std::size_t num_threads = helper::compute_num_threads_for_parallel_ip(_input_ctx);
    return split_and_join(coarsener, p_ctx, false, num_threads);
}

PartitionedGraph ParallelInitialPartitioner::partition_recursive(
    const ICoarsener* parent_coarsener, PartitionContext& p_ctx, const std::size_t num_threads) {
    const Graph* graph = parent_coarsener->coarsest_graph();

    if (num_threads == 1) { // base case: compute bipartition
        DBG << "Sequential base case";
        return helper::bipartition(graph, _input_ctx.partition.k, _input_ctx, _ip_m_ctx_pool);
    } else { // recursive / parallel case
        auto       coarsener = factory::create_coarsener(*graph, _input_ctx.coarsening);
        const bool shrunk    = helper::coarsen_once(coarsener.get(), graph, _input_ctx, p_ctx);

        PartitionedGraph p_graph = split_and_join(coarsener.get(), p_ctx, !shrunk, num_threads);

        // uncoarsen and refine
        p_graph       = helper::uncoarsen_once(coarsener.get(), std::move(p_graph), p_ctx);
        auto refiner  = factory::create_refiner(_input_ctx);
        auto balancer = factory::create_balancer(p_graph.graph(), p_ctx, _input_ctx.refinement);
        helper::refine(refiner.get(), balancer.get(), p_graph, p_ctx, _input_ctx.refinement);

        // extend partition
        const BlockID k_prime = helper::compute_k_for_n(p_graph.n(), _input_ctx);
        if (p_graph.k() < k_prime) {
            DBG << "Extend to " << k_prime << " ...";
            helper::extend_partition(p_graph, k_prime, _input_ctx, p_ctx, _ip_extraction_pool, _ip_m_ctx_pool);
        }

        return p_graph;
    }
}

PartitionedGraph ParallelInitialPartitioner::split_and_join(
    const ICoarsener* coarsener, const PartitionContext& p_ctx, const bool converged, const std::size_t num_threads) {
    const Graph*      graph            = coarsener->coarsest_graph();
    const std::size_t num_copies       = helper::compute_num_copies(_input_ctx, graph->n(), converged, num_threads);
    const std::size_t threads_per_copy = num_threads / num_copies;
    DBG << V(num_copies) << V(threads_per_copy) << V(converged) << V(num_threads) << V(graph->n());

    // parallel recursion
    tbb::task_group                   tg;
    scalable_vector<PartitionedGraph> p_graphs(num_copies);
    scalable_vector<PartitionContext> p_ctx_copies(num_copies, p_ctx);

    for (std::size_t copy = 0; copy < num_copies; ++copy) {
        tg.run([this, copy, coarsener, threads_per_copy, &p_graphs, &p_ctx_copies] { // must capture copy by value!
            p_graphs[copy] = partition_recursive(coarsener, p_ctx_copies[copy], threads_per_copy);
        });
    }
    tg.wait();

    // select best result
    const std::size_t best = helper::select_best(p_graphs, p_ctx);
    return std::move(p_graphs[best]);
}
} // namespace kaminpar::shm::partitioning
