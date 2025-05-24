#pragma once

#include <span>
#include <unordered_set>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/util/node_status.h"

namespace kaminpar::shm {

class MaxFlowAlgorithm {
public:
  struct Result {
    EdgeWeight flow_value;
    std::span<const EdgeWeight> flow;
  };

  virtual ~MaxFlowAlgorithm() = default;

  virtual void initialize(
      const CSRGraph &graph, std::span<const NodeID> reverse_edges, NodeID source, NodeID sink
  ) = 0;

  virtual void add_sources(std::span<const NodeID> sources) = 0;

  virtual void add_sinks(std::span<const NodeID> sinks) = 0;

  virtual void pierce_nodes(std::span<const NodeID> nodes, bool source_side) = 0;

  virtual Result compute_max_flow() = 0;

  virtual const NodeStatus &node_status() const = 0;
};

namespace debug {

[[nodiscard]] bool is_valid_flow(
    const CSRGraph &graph, const NodeStatus &node_status, std::span<const EdgeWeight> flow
);

[[nodiscard]] bool
is_max_flow(const CSRGraph &graph, const NodeStatus &node_status, std::span<const EdgeWeight> flow);

[[nodiscard]] EdgeWeight
flow_value(const CSRGraph &graph, const NodeStatus &node_status, std::span<const EdgeWeight> flow);

void print_flow(
    const CSRGraph &graph, const NodeStatus &node_status, std::span<const EdgeWeight> flow
);

} // namespace debug

} // namespace kaminpar::shm
