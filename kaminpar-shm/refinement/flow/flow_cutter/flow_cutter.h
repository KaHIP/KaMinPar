#pragma once

#include <memory>

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/flow_cutter/flow_cutter_algorithm.h"
#include "kaminpar-shm/refinement/flow/flow_network/border_region.h"
#include "kaminpar-shm/refinement/flow/flow_network/flow_network.h"
#include "kaminpar-shm/refinement/flow/max_flow/max_preflow_algorithm.h"
#include "kaminpar-shm/refinement/flow/piercing/piercing_heuristic.h"
#include "kaminpar-shm/refinement/flow/util/breadth_first_search.h"
#include "kaminpar-shm/refinement/flow/util/node_status.h"

#include "kaminpar-common/datastructures/marker.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

class FlowCutter : public FlowCutterAlgorithm {
  SET_DEBUG(false);

  static constexpr bool kSourceTag = true;
  static constexpr bool kSinkTag = false;

public:
  FlowCutter(const PartitionContext &p_ctx, const FlowCutterContext &fc_ctx);

  [[nodiscard]] Result compute_cut(
      const BorderRegion &border_region, const FlowNetwork &flow_network, bool run_sequentially
  ) override;

  void free() override;

private:
  void initialize(const BorderRegion &border_region, const FlowNetwork &flow_network);

  void run_flow_cutter(const BorderRegion &border_region, const FlowNetwork &flow_network);

  template <bool kCollectExcessNodes>
  void derive_source_side_cut(const FlowNetwork &flow_network, std::span<const EdgeWeight> flow);

  void derive_sink_side_cut(const FlowNetwork &flow_network, std::span<const EdgeWeight> flow);

  void update_border_nodes(bool source_side, const FlowNetwork &flow_network);

  void compute_moves(
      bool source_side, const BorderRegion &border_region, const FlowNetwork &flow_network
  );

  [[nodiscard]] MaxPreflowAlgorithm *max_preflow_algorithm();

private:
  const PartitionContext &_p_ctx;
  const FlowCutterContext &_fc_ctx;
  bool _run_sequentially;

  std::unique_ptr<MaxPreflowAlgorithm> _sequential_max_flow_algorithm;
  std::unique_ptr<MaxPreflowAlgorithm> _parallel_max_flow_algorithm;

  ScalableVector<NodeID> _source_side_border_nodes;
  ScalableVector<NodeID> _sink_side_border_nodes;

  ScalableVector<NodeID> _source_reachable_nodes;
  ScalableVector<NodeID> _sink_reachable_nodes;

  NodeWeight _source_reachable_weight;
  NodeWeight _sink_reachable_weight;

  BFSRunner _bfs_runner;
  Marker<> _source_reachable_nodes_marker;
  Marker<> _sink_reachable_nodes_marker;

  PiercingHeuristic _piercing_heuristic;
  Marker<> _source_side_piercing_node_candidates_marker;
  Marker<> _sink_side_piercing_node_candidates_marker;

  EdgeWeight _gain;
  bool _improve_balance;
  ScalableVector<Move> _moves;
};

} // namespace kaminpar::shm
