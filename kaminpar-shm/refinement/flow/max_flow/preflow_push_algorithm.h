#pragma once

#include <limits>
#include <queue>
#include <span>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/max_flow/max_preflow_algorithm.h"
#include "kaminpar-shm/refinement/flow/util/breadth_first_search.h"
#include "kaminpar-shm/refinement/flow/util/node_status.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

class PreflowPushAlgorithm : public MaxPreflowAlgorithm {
  SET_DEBUG(false);
  SET_STATISTICS(false);

  struct Statistics {
    std::size_t num_discharges;
    std::size_t num_global_relabels;

    void reset() {
      num_discharges = 0;
      num_global_relabels = 0;
    }

    void print() const {
      LOG_STATS << "Preflow-Push Algorithm:";
      LOG_STATS << "*  # num discharges: " << num_discharges;
      LOG_STATS << "*  # num global relabels: " << num_global_relabels;
    }
  };

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

  static constexpr bool kCollectActiveNodesTag = true;

public:
  PreflowPushAlgorithm(const PreflowPushContext &ctx);
  ~PreflowPushAlgorithm() override = default;

  void initialize(
      const CSRGraph &graph, std::span<const EdgeID> reverse_edges, NodeID source, NodeID sink
  ) override;

  void add_sources(std::span<const NodeID> sources) override;

  void add_sinks(std::span<const NodeID> sinks) override;

  void pierce_nodes(bool source_side, std::span<const NodeID> nodes) override;

  Result compute_max_preflow() override;

  std::span<const NodeID> excess_nodes() override;

  const NodeStatus &node_status() const override;

  void free() override;

private:
  void saturate_source_edges();

  template <bool kCollectActiveNodes = false> void global_relabel();

  void discharge(NodeID u);

  NodeID relabel(NodeID u);

private:
  const PreflowPushContext _ctx;
  Statistics _stats;

  const CSRGraph *_graph;
  std::span<const EdgeID> _reverse_edges;

  NodeStatus _node_status;
  ScalableVector<NodeID> _excess_nodes;

  EdgeWeight _flow_value;
  StaticArray<EdgeWeight> _flow;

  bool _force_global_relabel;
  GlobalRelabelingThreshold _grt;
  BFSRunner _bfs_runner;

  ScalableVector<NodeID> _nodes_to_desaturate;

  StaticArray<NodeID> _cur_edge_offsets;
  StaticArray<EdgeWeight> _excess;
  StaticArray<NodeID> _heights;
  std::queue<NodeID> _active_nodes;
};

} // namespace kaminpar::shm
