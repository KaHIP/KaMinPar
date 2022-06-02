/*******************************************************************************
 * @file:   deep_mgp.h
 *
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 * @brief:  Partitioning scheme using deep multilevel graph partitioning.
 ******************************************************************************/
#pragma once

#include "dkaminpar/context.h"
#include "dkaminpar/datastructure/distributed_graph.h"

namespace dkaminpar {
class DeepMGPPartitioningScheme {
public:
    DeepMGPPartitioningScheme(const DistributedGraph& input_graph, const Context& input_ctx);

    DeepMGPPartitioningScheme(const DeepMGPPartitioningScheme&) = delete;
    DeepMGPPartitioningScheme& operator=(const DeepMGPPartitioningScheme&) = delete;

    DeepMGPPartitioningScheme(DeepMGPPartitioningScheme&&) noexcept = default;
    DeepMGPPartitioningScheme& operator=(DeepMGPPartitioningScheme&&) = delete;

    DistributedPartitionedGraph partition();

private:
    const DistributedGraph& _input_graph;
    const Context&          _input_ctx;
};
} // namespace dkaminpar
