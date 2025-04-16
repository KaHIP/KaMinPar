#pragma once

#include <span>
#include <unordered_set>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {

class MaxFlowAlgorithm {
public:
  virtual ~MaxFlowAlgorithm() = default;

  virtual void initialize(const CSRGraph &graph) = 0;

  virtual std::span<const EdgeWeight> compute_max_flow(
      const std::unordered_set<NodeID> &sources, const std::unordered_set<NodeID> &sinks
  ) = 0;
};

namespace debug {

[[nodiscard]] bool are_terminals_disjoint(
    const std::unordered_set<NodeID> &sources, const std::unordered_set<NodeID> &sinks
);

[[nodiscard]] bool is_valid_flow(
    const CSRGraph &graph,
    const std::unordered_set<NodeID> &sources,
    const std::unordered_set<NodeID> &sinks,
    std::span<const EdgeWeight> flow
);

[[nodiscard]] bool is_max_flow(
    const CSRGraph &graph,
    const std::unordered_set<NodeID> &sources,
    const std::unordered_set<NodeID> &sinks,
    std::span<const EdgeWeight> flow
);

void print_flow(
    const CSRGraph &graph,
    const std::unordered_set<NodeID> &sources,
    const std::unordered_set<NodeID> &sinks,
    std::span<const EdgeWeight> flow
);

} // namespace debug

} // namespace kaminpar::shm
