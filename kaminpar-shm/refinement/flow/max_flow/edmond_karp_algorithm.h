#pragma once

#include <span>
#include <unordered_set>
#include <utility>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/max_flow/max_flow_algorithm.h"

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

  void initialize(const CSRGraph &graph) override;

  std::span<const EdgeWeight> compute_max_flow(
      const std::unordered_set<NodeID> &sources, const std::unordered_set<NodeID> &sinks
  ) override;

private:
  std::pair<NodeID, EdgeWeight> find_augmenting_path(
      const std::unordered_set<NodeID> &sources, const std::unordered_set<NodeID> &sinks
  );

  void augment_flow(NodeID sink, EdgeWeight net_flow);

private:
  const CSRGraph *_graph;
  StaticArray<EdgeID> _reverse_edge_index;

  StaticArray<EdgeWeight> _flow;
  StaticArray<PathEdge> _predecessor;
};

} // namespace kaminpar::shm
