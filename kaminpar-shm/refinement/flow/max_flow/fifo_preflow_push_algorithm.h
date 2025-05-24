#pragma once

#include <limits>
#include <queue>
#include <span>

#include "kaminpar.h"

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/max_flow/max_flow_algorithm.h"
#include "kaminpar-shm/refinement/flow/util/node_status.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

class FIFOPreflowPushAlgorithm : public MaxFlowAlgorithm {
  SET_DEBUG(false);

  class GlobalRelabelingThreshold {
  public:
    GlobalRelabelingThreshold() : _threshold(std::numeric_limits<std::size_t>::max()), _work(0) {}

    GlobalRelabelingThreshold(
        const NodeID num_nodes, const EdgeID num_edges, const double frequency
    )
        : _threshold((num_nodes + num_edges) / frequency),
          _work(0) {}

    bool is_reached() const {
      return _work >= _threshold;
    }

    void add_work(const std::size_t work) {
      _work += work;
    }

    void clear() {
      _work = 0;
    }

  private:
    std::size_t _threshold;
    std::size_t _work;
  };

public:
  FIFOPreflowPushAlgorithm(const FIFOPreflowPushContext &ctx);
  ~FIFOPreflowPushAlgorithm() override = default;

  void initialize(
      const CSRGraph &graph, std::span<const NodeID> reverse_edges, NodeID source, NodeID sink
  ) override;

  void add_sources(std::span<const NodeID> sources) override;

  void add_sinks(std::span<const NodeID> sinks) override;

  void pierce_nodes(std::span<const NodeID> nodes, bool source_side) override;

  Result compute_max_flow() override;

  const NodeStatus &node_status() const override;

private:
  template <bool kSetSourceHeight = false>
  void saturate_source_edges(std::span<const NodeID> sources);

  void global_relabel();

  void discharge(NodeID u);

  template <bool kFromSource = false>
  void push(NodeID from, NodeID to, EdgeID e, EdgeWeight residual_capacity);

  NodeID relabel(NodeID u);

private:
  const FIFOPreflowPushContext _ctx;

  const CSRGraph *_graph;
  std::span<const NodeID> _reverse_edges;

  NodeStatus _node_status;

  EdgeWeight _flow_value;
  StaticArray<EdgeWeight> _flow;

  GlobalRelabelingThreshold _grt;
  StaticArray<EdgeWeight> _excess;
  StaticArray<NodeID> _cur_edge_offsets;
  StaticArray<NodeID> _heights;
  std::queue<NodeID> _active_nodes;
};

} // namespace kaminpar::shm
