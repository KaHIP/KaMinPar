/*******************************************************************************
 * @file:   deep_mgp.cc
 *
 * @author: Daniel Seemaier
 * @date:   28.04.2022
 * @brief:  Partitioning scheme using deep multilevel graph partitioning.
 ******************************************************************************/
#include "dkaminpar/partitioning_scheme/deep_mgp.h"

#include "dkaminpar/datastructure/distributed_graph.h"
#include "dkaminpar/partitioning_scheme/kway.h"

namespace dkaminpar {
DeepMGPPartitioningScheme::DeepMGPPartitioningScheme(const DistributedGraph& input_graph, const Context& input_ctx)
    : _input_graph(input_graph),
      _input_ctx(input_ctx) {}

DistributedPartitionedGraph DeepMGPPartitioningScheme::partition() {
    return KWayPartitioningScheme(_input_graph, _input_ctx).partition();
}
} // namespace dkaminpar
