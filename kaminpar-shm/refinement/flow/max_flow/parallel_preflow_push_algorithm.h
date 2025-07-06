#pragma once

#include <cstdint>
#include <limits>
#include <span>

#include <oneapi/tbb/concurrent_vector.h>
#include <oneapi/tbb/enumerable_thread_specific.h>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/max_flow/max_preflow_algorithm.h"
#include "kaminpar-shm/refinement/flow/util/breadth_first_search.h"
#include "kaminpar-shm/refinement/flow/util/buffered_vector.h"
#include "kaminpar-shm/refinement/flow/util/node_status.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

class ParallelPreflowPushAlgorithm : public MaxPreflowAlgorithm {
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

  static constexpr std::uint8_t kNotModifiedState = 0;
  static constexpr std::uint8_t kPushedState = 1;
  static constexpr std::uint8_t kRelabeledState = 2;

public:
  ParallelPreflowPushAlgorithm(const PreflowPushContext &ctx);
  ~ParallelPreflowPushAlgorithm() override = default;

  void initialize(
      const CSRGraph &graph, std::span<const NodeID> reverse_edges, NodeID source, NodeID sink
  ) override;

  void add_sources(std::span<const NodeID> sources) override;

  void add_sinks(std::span<const NodeID> sinks) override;

  void pierce_nodes(bool source_side, std::span<const NodeID> nodes) override;

  Result compute_max_preflow() override;

  std::span<const NodeID> excess_nodes() override;

  const NodeStatus &node_status() const override;

private:
  void saturate_source_edges(std::span<const NodeID> sources);

  template <bool kCollectActiveNodes = false> void global_relabel();

  void discharge_active_nodes();

  void apply_updates();

  void discharge(NodeID u, BufferedVector<NodeID>::Buffer next_active_nodes);

  NodeID relabel(NodeID u);

  bool update_active_node(NodeID u, std::uint8_t desired_state);

private:
  const PreflowPushContext _ctx;

  const CSRGraph *_graph;
  std::span<const NodeID> _reverse_edges;

  NodeStatus _node_status;
  ScalableVector<NodeID> _excess_nodes;

  EdgeWeight _flow_value;
  StaticArray<EdgeWeight> _flow;

  std::size_t _round;

  GlobalRelabelingThreshold _grt;
  tbb::enumerable_thread_specific<std::size_t> _work_ets;
  ParallelBFSRunner _parallel_bfs_runner;

  StaticArray<NodeID> _last_touched;
  StaticArray<NodeID> _cur_edge_offsets;
  StaticArray<std::uint8_t> _active_node_state;

  StaticArray<EdgeWeight> _excess;
  StaticArray<EdgeWeight> _excess_delta;

  StaticArray<NodeID> _heights;
  StaticArray<NodeID> _next_heights;

  BufferedVector<NodeID> _active_nodes;
  BufferedVector<NodeID> _next_active_nodes;
};

} // namespace kaminpar::shm
