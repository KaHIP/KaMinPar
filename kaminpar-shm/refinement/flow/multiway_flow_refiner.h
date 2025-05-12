/*******************************************************************************
 * Multi-way flow refiner.
 *
 * @file:   multiway_flow_refiner.h
 * @author: Daniel Salwasser
 * @date:   10.04.2025
 ******************************************************************************/
#pragma once

#include <unordered_map>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/multiway_cut/multiway_cut_algorithm.h"
#include "kaminpar-shm/refinement/flow/util/border_region.h"
#include "kaminpar-shm/refinement/refiner.h"

#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

class MultiwayFlowRefiner : public Refiner {
  SET_DEBUG(true);

  struct FlowNetwork {
    CSRGraph graph;
    std::unordered_map<NodeID, NodeID> global_to_local_mapping;
  };

  struct Cut {
    NodeWeight weight;
    std::unordered_set<NodeID> nodes;
  };

public:
  MultiwayFlowRefiner(const Context &ctx);
  ~MultiwayFlowRefiner() override;

  MultiwayFlowRefiner(const MultiwayFlowRefiner &) = delete;
  MultiwayFlowRefiner &operator=(const MultiwayFlowRefiner &) = delete;

  MultiwayFlowRefiner(MultiwayFlowRefiner &&) noexcept = default;
  MultiwayFlowRefiner &operator=(MultiwayFlowRefiner &&) noexcept = default;

  [[nodiscard]] std::string name() const override;

  void initialize(const PartitionedGraph &p_graph) override;

  bool refine(PartitionedGraph &p_graph, const PartitionContext &p_ctx) override;

private:
  bool refine(PartitionedGraph &p_graph, const CSRGraph &graph, const PartitionContext &p_ctx);

  std::vector<BorderRegion> compute_border_regions(const PartitionContext &p_ctx) const;

  void expand_border_region(BorderRegion &border_region) const;

  FlowNetwork construct_flow_network(const std::vector<BorderRegion> &border_regions);

  Cut compute_cut_nodes(
      const CSRGraph &graph,
      const std::unordered_set<NodeID> &terminals,
      const std::unordered_set<EdgeID> &cut_edges
  );

private:
  const MultiwayFlowRefinementContext &_f_ctx;

  PartitionedGraph *_p_graph;
  const CSRGraph *_graph;

  std::unique_ptr<MultiwayCutAlgorithm> _multiway_cut_algorithm;
};

} // namespace kaminpar::shm
