#pragma once

#include <utility>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/max_flow/max_flow_algorithm.h"
#include "kaminpar-shm/refinement/flow/util/node_status.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

class EdmondsKarpAlgorithm : public MaxFlowAlgorithm {
  SET_DEBUG(false);

  struct PathEdge {
    NodeID from;
    EdgeID edge;
  };

public:
  EdmondsKarpAlgorithm() = default;
  ~EdmondsKarpAlgorithm() override = default;

  void initialize(
      const CSRGraph &graph, std::span<const NodeID> reverse_edges, NodeID source, NodeID sink
  ) override;

  void add_sources(std::span<const NodeID> sources) override;

  void add_sinks(std::span<const NodeID> sinks) override;

  void pierce_nodes(std::span<const NodeID> nodes, bool source_side) override;

  Result compute_max_flow() override;

  const NodeStatus &node_status() const override;

private:
  std::pair<NodeID, EdgeWeight> find_augmenting_path();

  void augment_flow(NodeID sink, EdgeWeight net_flow);

private:
  const CSRGraph *_graph;
  std::span<const NodeID> _reverse_edges;

  NodeStatus _node_status;

  EdgeWeight _flow_value;
  StaticArray<EdgeWeight> _flow;

  StaticArray<PathEdge> _predecessor;
};

} // namespace kaminpar::shm
