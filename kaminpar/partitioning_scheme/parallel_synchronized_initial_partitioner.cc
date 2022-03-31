/*******************************************************************************
 * @file:   parallel_synchronized_initial_partitioner.cc
 *
 * @author: Daniel Seemaier
 * @date:   21.09.21
 * @brief:
 ******************************************************************************/
#include "kaminpar/partitioning_scheme/parallel_synchronized_initial_partitioner.h"

namespace kaminpar::partitioning {
ParallelSynchronizedInitialPartitioner::ParallelSynchronizedInitialPartitioner(
    const Context& input_ctx, GlobalInitialPartitionerMemoryPool& ip_m_ctx_pool,
    TemporaryGraphExtractionBufferPool& ip_extraction_pool)
    : _input_ctx{input_ctx},
      _ip_m_ctx_pool{ip_m_ctx_pool},
      _ip_extraction_pool{ip_extraction_pool} {}

PartitionedGraph
ParallelSynchronizedInitialPartitioner::partition(const ICoarsener* coarsener, const PartitionContext& p_ctx) {
    const std::size_t num_threads = helper::compute_num_threads_for_parallel_ip(_input_ctx);

    std::vector<std::vector<std::unique_ptr<ICoarsener>>> coarseners(1);
    std::vector<PartitionContext>                         current_p_ctxs;
    coarseners[0].push_back(duplicate_coarsener(coarsener));
    current_p_ctxs.push_back(p_ctx);

    std::size_t       num_current_threads = num_threads;
    std::size_t       num_current_copies  = 1;
    std::atomic<bool> converged           = false;

    std::vector<std::size_t> num_local_copies_record;
    while (num_current_copies < num_threads) {
        const NodeID      n                = coarseners.back()[0]->coarsest_graph()->n();
        const std::size_t num_local_copies = helper::compute_num_copies(_input_ctx, n, converged, num_current_threads);
        num_local_copies_record.push_back(num_local_copies);

        // create coarseners and partition contexts for next coarsening iteration
        coarseners.emplace_back(num_current_copies * num_local_copies);
        auto&                         next_coarseners    = coarseners.back();
        auto&                         current_coarseners = coarseners[coarseners.size() - 2];
        std::vector<PartitionContext> next_p_ctxs(num_current_copies * num_local_copies);

        tbb::parallel_for(static_cast<std::size_t>(0), num_current_copies, [&](const std::size_t i) {
            const std::size_t next_i = i * num_local_copies;
            for (std::size_t next_j = next_i; next_j < next_i + num_local_copies; ++next_j) {
                DBG << "Duplicate " << i << " --> " << next_j;
                next_coarseners[next_j] = duplicate_coarsener(current_coarseners[i].get());
                next_p_ctxs[next_j]     = current_p_ctxs[i];
            }
        });

        std::swap(current_p_ctxs, next_p_ctxs);

        num_current_threads /= num_local_copies;
        num_current_copies *= num_local_copies;

        // perform coarsening iteration, converge if all coarseners converged
        converged = true;
        tbb::parallel_for(static_cast<std::size_t>(0), num_current_copies, [&](const std::size_t i) {
            const bool shrunk = helper::coarsen_once(
                next_coarseners[i].get(), next_coarseners[i]->coarsest_graph(), _input_ctx, current_p_ctxs[i]);
            if (shrunk) {
                converged = false;
            }
        });
    }

    // perform initial bipartition on every graph
    std::vector<PartitionedGraph> current_p_graphs(num_threads);
    tbb::parallel_for(static_cast<std::size_t>(0), num_threads, [&](const std::size_t i) {
        auto&        current_coarseners = coarseners.back();
        const Graph* graph              = current_coarseners[i]->coarsest_graph();
        current_p_graphs[i] = helper::bipartition(graph, _input_ctx.partition.k, _input_ctx, _ip_m_ctx_pool);
    });

    // uncoarsen and join graphs
    while (!num_local_copies_record.empty()) {
        const std::size_t num_local_copies = num_local_copies_record.back();
        num_local_copies_record.pop_back();

        auto& current_coarseners = coarseners.back();

        // uncoarsen and refine
        tbb::parallel_for(static_cast<std::size_t>(0), num_current_copies, [&](const std::size_t i) {
            auto& p_graph   = current_p_graphs[i];
            auto& coarsener = current_coarseners[i];
            auto& p_ctx     = current_p_ctxs[i];

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
        });

        num_current_copies /= num_local_copies;
        num_current_threads *= num_local_copies;

        std::vector<PartitionContext> next_p_ctxs(num_current_copies);
        std::vector<PartitionedGraph> next_p_graphs(num_current_copies);

        tbb::parallel_for(static_cast<std::size_t>(0), num_current_copies, [&](const std::size_t i) {
            // join
            const std::size_t start_pos = i * num_local_copies;
            PartitionContext& p_ctx     = current_p_ctxs[start_pos];
            const std::size_t pos       = helper::select_best(
                      current_p_graphs.begin() + start_pos, current_p_graphs.begin() + start_pos + num_local_copies, p_ctx);
            PartitionedGraph& p_graph = current_p_graphs[start_pos + pos];

            // store
            next_p_ctxs[i]   = std::move(p_ctx);
            next_p_graphs[i] = std::move(p_graph);
        });

        std::swap(current_p_ctxs, next_p_ctxs);
        std::swap(current_p_graphs, next_p_graphs);
        coarseners.pop_back();
    }

    ALWAYS_ASSERT(coarseners.size() == 1);
    ALWAYS_ASSERT(&(current_p_graphs.front().graph()) == coarsener->coarsest_graph());
    return std::move(current_p_graphs.front());
}

std::unique_ptr<ICoarsener> ParallelSynchronizedInitialPartitioner::duplicate_coarsener(const ICoarsener* coarsener) {
    return factory::create_coarsener(*coarsener->coarsest_graph(), _input_ctx.coarsening);
}
} // namespace kaminpar::partitioning
