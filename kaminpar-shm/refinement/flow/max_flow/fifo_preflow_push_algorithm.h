#pragma once

#include <limits>
#include <queue>
#include <span>
#include <unordered_set>
#include "kaminpar.h"

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/max_flow/max_flow_algorithm.h"

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

  void initialize(const CSRGraph &graph) override;

  std::span<const EdgeWeight> compute_max_flow(
      const std::unordered_set<NodeID> &sources, const std::unordered_set<NodeID> &sinks
  ) override;

private:
  void saturate_source_edges();

  void global_relabel();

  void discharge(NodeID u);

  void push(NodeID from, NodeID to, EdgeID e, EdgeWeight residual_capacity);

  NodeID relabel(NodeID u);

private:
  const FIFOPreflowPushContext _ctx;

  const CSRGraph *_graph;
  StaticArray<EdgeID> _reverse_edge_index;

  const std::unordered_set<NodeID> *_sources;
  const std::unordered_set<NodeID> *_sinks;

  GlobalRelabelingThreshold _grt;
  StaticArray<EdgeWeight> _flow;
  StaticArray<NodeID> _heights;
  StaticArray<EdgeWeight> _excess;
  StaticArray<NodeID> _cur_edge_offsets;
  std::queue<NodeID> _active_nodes;
};

} // namespace kaminpar::shm
