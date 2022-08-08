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
    inline Coarsener* get_current_coarsener();

    const DistributedGraph& _input_graph;
    const Context&          _input_ctx;

    std::stack<Coarsener> _coarseners;
};
} // namespace kaminpar::dist
