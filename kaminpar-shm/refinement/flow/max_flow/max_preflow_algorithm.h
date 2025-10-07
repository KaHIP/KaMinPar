#pragma once

#include <span>
#include <utility>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/util/node_status.h"

namespace kaminpar::shm {

class MaxPreflowAlgorithm {
public:
  using Result = std::pair<EdgeWeight, std::span<const EdgeWeight>>;

  virtual ~MaxPreflowAlgorithm() = default;

  virtual void initialize(
      const CSRGraph &graph, std::span<const EdgeID> reverse_edges, NodeID source, NodeID sink
  ) = 0;

  virtual void add_sources(std::span<const NodeID> sources) = 0;

  virtual void add_sinks(std::span<const NodeID> sinks) = 0;

  virtual void pierce_nodes(bool source_side, std::span<const NodeID> nodes) = 0;

  virtual Result compute_max_preflow() = 0;

  virtual std::span<const NodeID> excess_nodes() = 0;

  virtual const NodeStatus &node_status() const = 0;

  virtual void free() = 0;
};

namespace debug {

[[nodiscard]] bool is_valid_labeling(
    const CSRGraph &graph,
    const NodeStatus &node_status,
    std::span<const EdgeWeight> flow,
    std::span<const NodeID> labeling
);

[[nodiscard]] bool is_valid_flow(
    const CSRGraph &graph, const NodeStatus &node_status, std::span<const EdgeWeight> flow
);

[[nodiscard]] bool is_valid_preflow(
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
