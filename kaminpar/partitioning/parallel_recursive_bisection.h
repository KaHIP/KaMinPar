/*******************************************************************************
 * @file:   parallel_recursive_bisection.h
 * @author: Daniel Seemaier
 * @date:   21.09.2021
 * @brief:
 ******************************************************************************/
#pragma once

#include <tbb/enumerable_thread_specific.h>

#include "kaminpar/coarsening/label_propagation_clustering.h"
#include "kaminpar/context.h"
#include "kaminpar/datastructures/graph.h"
#include "kaminpar/factories.h"
#include "kaminpar/graphutils/graph_extraction.h"
#include "kaminpar/initial_partitioning/initial_partitioning_facade.h"
#include "kaminpar/initial_partitioning/pool_bipartitioner.h"
#include "kaminpar/partitioning/helper.h"

#include "common/console_io.h"

namespace kaminpar::shm::partitioning {
class ParallelRecursiveBisection {
    SET_DEBUG(false);
    SET_STATISTICS(false);

public:
    ParallelRecursiveBisection(const Graph& input_graph, const Context& input_ctx);

    ParallelRecursiveBisection(const ParallelRecursiveBisection&)            = delete;
    ParallelRecursiveBisection& operator=(const ParallelRecursiveBisection&) = delete;
    ParallelRecursiveBisection(ParallelRecursiveBisection&&)                 = delete;
    ParallelRecursiveBisection& operator=(ParallelRecursiveBisection&&)      = delete;

    PartitionedGraph partition();

private:
    PartitionedGraph uncoarsen(PartitionedGraph p_graph, bool& refined);

    inline PartitionedGraph uncoarsen_once(PartitionedGraph p_graph) {
        return helper::uncoarsen_once(_coarsener.get(), std::move(p_graph), _current_p_ctx);
    }

    inline void refine(PartitionedGraph& p_graph) {
        helper::refine(_refiner.get(), _balancer.get(), p_graph, _current_p_ctx, _input_ctx.refinement);
    }

    inline void extend_partition(PartitionedGraph& p_graph, const BlockID k_prime) {
        helper::extend_partition(
            p_graph, k_prime, _input_ctx, _current_p_ctx, _subgraph_memory, _ip_extraction_pool, _ip_m_ctx_pool
        );
    }

    const Graph*     coarsen();
    NodeID           initial_partition_threshold();
    PartitionedGraph initial_partition(const Graph* graph);
    PartitionedGraph parallel_initial_partition(const Graph* /* use _coarsener */);
    PartitionedGraph sequential_initial_partition(const Graph* graph);
    void             print_statistics();

    const Graph&     _input_graph;
    const Context&   _input_ctx;
    PartitionContext _current_p_ctx;

    // Coarsening
    std::unique_ptr<ICoarsener> _coarsener;

    // Refinement
    std::unique_ptr<IRefiner>  _refiner;
    std::unique_ptr<IBalancer> _balancer;

    // Initial partitioning -> subgraph extraction
    graph::SubgraphMemory              _subgraph_memory;
    TemporaryGraphExtractionBufferPool _ip_extraction_pool;

    // Initial partitioning
    GlobalInitialPartitionerMemoryPool _ip_m_ctx_pool;
};
} // namespace kaminpar::shm::partitioning
