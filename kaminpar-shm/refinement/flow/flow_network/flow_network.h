#pragma once


#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/flow_network/border_region.h"

#include "kaminpar-common/datastructures/dynamic_map.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

struct FlowNetwork {
  NodeID source;
  NodeID sink;

  CSRGraph graph;
  StaticArray<EdgeID> reverse_edges;

  DynamicRememberingFlatMap<NodeID, NodeID> global_to_local_mapping;
  DynamicRememberingFlatMap<NodeID, NodeID> local_to_global_mapping;
};

[[nodiscard]] std::pair<FlowNetwork, EdgeWeight> construct_flow_network(
    const CSRGraph &graph,
    const PartitionedCSRGraph &p_graph,
    const BlockWeight block1_weight,
    const BlockWeight block2_weight,
    const BorderRegion &border_region1,
    const BorderRegion &border_region2
);

[[nodiscard]] std::pair<FlowNetwork, EdgeWeight> parallel_construct_flow_network(
    const CSRGraph &graph,
    const PartitionedCSRGraph &p_graph,
    const BlockWeight block1_weight,
    const BlockWeight block2_weight,
    const BorderRegion &border_region1,
    const BorderRegion &border_region2
);

} // namespace kaminpar::shm
