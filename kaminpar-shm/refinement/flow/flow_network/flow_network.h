#pragma once

#include <span>
#include <unordered_map>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/flow_network/border_region.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

struct FlowNetwork {
  NodeID source;
  NodeID sink;

  CSRGraph graph;
  StaticArray<NodeID> reverse_edges;
  std::unordered_map<NodeID, NodeID> global_to_local_mapping;
};

[[nodiscard]] std::pair<FlowNetwork, EdgeWeight> construct_flow_network(
    const CSRGraph &_graph,
    const BorderRegion &_border_region1,
    const BorderRegion &_border_region2,
    std::span<const BlockID> _partition,
    std::span<const BlockWeight> _block_weights
);

[[nodiscard]] std::pair<FlowNetwork, EdgeWeight> parallel_construct_flow_network(
    const CSRGraph &_graph,
    const BorderRegion &_border_region1,
    const BorderRegion &_border_region2,
    std::span<const BlockID> _partition,
    std::span<const BlockWeight> _block_weights
);

} // namespace kaminpar::shm
