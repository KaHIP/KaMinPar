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

  BlockWeight block1_weight;
  BlockWeight block2_weight;
  EdgeWeight cut_value;
};

class FlowNetworkConstructor {
public:
  FlowNetworkConstructor(
      const FlowNetworkConstructionContext &c_ctx,
      const PartitionedCSRGraph &p_graph,
      const CSRGraph &graph
  );

  FlowNetworkConstructor(FlowNetworkConstructor &&) noexcept = default;
  FlowNetworkConstructor &operator=(FlowNetworkConstructor &&) noexcept = delete;

  FlowNetworkConstructor(const FlowNetworkConstructor &) = delete;
  FlowNetworkConstructor &operator=(const FlowNetworkConstructor &) = delete;

  [[nodiscard]] FlowNetwork construct_flow_network(
      const BorderRegion &border_region,
      BlockWeight block1_weight,
      BlockWeight block2_weight,
      bool run_sequentially
  );

private:
  [[nodiscard]] FlowNetwork sequential_construct_flow_network(
      const BorderRegion &border_region, BlockWeight block1_weight, BlockWeight block2_weight
  );

  [[nodiscard]] FlowNetwork parallel_construct_flow_network(
      const BorderRegion &border_region, BlockWeight block1_weight, BlockWeight block2_weight
  );

private:
  const FlowNetworkConstructionContext &_c_ctx;

  const PartitionedCSRGraph &_p_graph;
  const CSRGraph &_graph;
};

} // namespace kaminpar::shm
