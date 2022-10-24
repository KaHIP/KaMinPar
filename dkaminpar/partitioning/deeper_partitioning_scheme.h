/*******************************************************************************
 * @file:   deep_partitioning.h
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 * @brief:  Deep multilevel graph partitioning scheme.
 ******************************************************************************/
#pragma once

#include <list>
#include <stack>

#include "dkaminpar/coarsening/coarsener.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructures/distributed_graph.h"

namespace kaminpar::dist {
class DeeperPartitioningScheme {
public:
    DeeperPartitioningScheme(const DistributedGraph& input_graph, const Context& input_ctx);

    DeeperPartitioningScheme(const DeeperPartitioningScheme&)            = delete;
    DeeperPartitioningScheme& operator=(const DeeperPartitioningScheme&) = delete;

    DeeperPartitioningScheme(DeeperPartitioningScheme&&) noexcept   = default;
    DeeperPartitioningScheme& operator=(DeeperPartitioningScheme&&) = delete;

    DistributedPartitionedGraph partition();

private:
    void print_coarsening_level(GlobalNodeWeight max_cluster_weight) const;
    void print_coarsening_converged() const;
    void print_coarsening_terminated(GlobalNodeID desired_num_nodes) const;
    void
    print_initial_partitioning_result(const DistributedPartitionedGraph& p_graph, const PartitionContext& p_ctx) const;

    inline Coarsener* get_current_coarsener();
    const Coarsener*  get_current_coarsener() const;

    const DistributedGraph& _input_graph;
    const Context&          _input_ctx;

    std::list<DistributedGraph> _replicated_graphs;
    std::stack<Coarsener>       _coarseners;
};
} // namespace kaminpar::dist
