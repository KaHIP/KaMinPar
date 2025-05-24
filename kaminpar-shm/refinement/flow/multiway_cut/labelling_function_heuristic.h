#pragma once

#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/max_flow/max_flow_algorithm.h"
#include "kaminpar-shm/refinement/flow/multiway_cut/multiway_cut_algorithm.h"

#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

class LabellingFunctionHeuristic : public MultiwayCutAlgorithm {
  SET_DEBUG(true);

  struct FlowNetwork {
    NodeID source;
    NodeID sink;

    CSRGraph graph;
  };

public:
  LabellingFunctionHeuristic(const LabellingFunctionHeuristicContext &ctx);

  [[nodiscard]] MultiwayCutAlgorithm::Result compute(
      const CSRGraph &graph, const std::vector<std::unordered_set<NodeID>> &terminal_sets
  ) override;

  [[nodiscard]] MultiwayCutAlgorithm::Result compute(
      const PartitionedCSRGraph &p_graph,
      const CSRGraph &graph,
      const std::vector<std::unordered_set<NodeID>> &terminal_sets
  ) override;

private:
  const LabellingFunctionHeuristicContext &_ctx;

  const CSRGraph *_graph;
  StaticArray<EdgeID> _reverse_edge_index;

  std::unordered_set<NodeID> _terminals;
  std::unordered_map<NodeID, BlockID> _terminal_labels;

  EdgeWeight _labelling_function_cost;
  StaticArray<BlockID> _labelling_function;

  std::unique_ptr<MaxFlowAlgorithm> _max_flow_algorithm;

private:
  void initialize_labelling_function();

  void initialize_labelling_function(const PartitionedCSRGraph &p_graph);

  void improve_labelling_function();

  EdgeWeight compute_labelling_function_cost() const;

  FlowNetwork construct_flow_network(BlockID label) const;

  void derive_labelling_function(
      BlockID label, const FlowNetwork &flow_network, std::span<const EdgeWeight> flow
  );

  std::unordered_set<EdgeID> derive_cut_edges() const;

  std::unordered_set<NodeID> static compute_cut_nodes(
      const CSRGraph &graph, const NodeID terminal, std::span<const EdgeWeight> flow
  );

  bool is_valid_labelling_function() const;
};

} // namespace kaminpar::shm
