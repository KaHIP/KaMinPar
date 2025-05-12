#pragma once

#include <memory>
#include <span>
#include <unordered_set>
#include <vector>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/max_flow/max_flow_algorithm.h"
#include "kaminpar-shm/refinement/flow/multiway_cut/multiway_cut_algorithm.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

class IsolatingCutHeuristic : public MultiwayCutAlgorithm {
  struct Cut {
    EdgeWeight value;
    std::unordered_set<EdgeID> edges;
  };

public:
  using MultiwayCutAlgorithm::compute;

  IsolatingCutHeuristic(const IsolatingCutHeuristicContext &ctx);

  [[nodiscard]] MultiwayCutAlgorithm::Result compute(
      const CSRGraph &graph, const std::vector<std::unordered_set<NodeID>> &terminal_sets
  ) override;

private:
  Cut compute_cut(const std::unordered_set<NodeID> &terminals, std::span<const EdgeWeight> flow);

  EdgeWeight compute_cut_value(const std::unordered_set<EdgeID> &cut_edges);

private:
  const IsolatingCutHeuristicContext &_ctx;

  const CSRGraph *_graph;
  StaticArray<EdgeID> _reverse_edge_index;

  std::unique_ptr<MaxFlowAlgorithm> _max_flow_algorithm;
};

} // namespace kaminpar::shm
