/*******************************************************************************
 * @file:   deep_partitioning_scheme.h
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 * @brief:  Deep multilevel graph partitioning scheme.
 ******************************************************************************/
#pragma once

#include <stack>

#include "dkaminpar/coarsening/coarsener.h"
#include "dkaminpar/context.h"
#include "dkaminpar/datastructure/distributed_graph.h"

namespace kaminpar::dist {
class DeepPartitioningScheme {
public:
    DeepPartitioningScheme(const DistributedGraph& input_graph, const Context& input_ctx);

    DeepPartitioningScheme(const DeepPartitioningScheme&)            = delete;
    DeepPartitioningScheme& operator=(const DeepPartitioningScheme&) = delete;

    DeepPartitioningScheme(DeepPartitioningScheme&&) noexcept   = default;
    DeepPartitioningScheme& operator=(DeepPartitioningScheme&&) = delete;

    DistributedPartitionedGraph partition();

private:
    void print_coarsening_level(GlobalNodeWeight max_cluster_weight) const;
    void print_coarsening_converged() const;
    void print_coarsening_terminated(GlobalNodeID desired_num_nodes) const;
    void print_initial_partitioning_result(const DistributedPartitionedGraph& p_graph) const;

    inline Coarsener* get_current_coarsener();
    const Coarsener*  get_current_coarsener() const;

    const DistributedGraph& _input_graph;
    const Context&          _input_ctx;

    std::stack<Coarsener> _coarseners;
};
} // namespace kaminpar::dist
