#pragma once

#include <limits>
#include <span>

#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>

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
  SET_STATISTICS(false);

  struct Statistics {
    std::size_t num_rounds;
    std::size_t num_sequential_rounds;
    std::size_t num_parallel_rounds;
    std::size_t num_discharges;
    std::size_t num_parallel_discharges;
    std::size_t num_global_relabels;
    tbb::enumerable_thread_specific<std::size_t> num_win_conflicts_ets;
    tbb::enumerable_thread_specific<std::size_t> num_invalid_labels_ets;

    void reset() {
      num_rounds = 0;
      num_sequential_rounds = 0;
      num_parallel_rounds = 0;
      num_discharges = 0;
      num_parallel_discharges = 0;
      num_global_relabels = 0;
      num_win_conflicts_ets.clear();
      num_invalid_labels_ets.clear();
    }

    void print() const {
      LOG_STATS << "Parallel Preflow-Push Algorithm:";
      LOG_STATS << "*  # num rounds (sequential / parallel): " << num_rounds << " ("
                << num_sequential_rounds << " / " << num_parallel_rounds << ")";
      LOG_STATS << "*  # num discharges (sequential / parallel): " << num_discharges << " ("
                << (num_discharges - num_parallel_discharges) << " / " << num_parallel_discharges
                << ")";
      LOG_STATS << "*  # num global relabels: " << num_global_relabels;
      LOG_STATS << "*  # num win conflicts: "
                << std::accumulate(num_win_conflicts_ets.begin(), num_win_conflicts_ets.end(), 0);
      LOG_STATS << "*  # num invalid labels: "
                << std::accumulate(num_invalid_labels_ets.begin(), num_invalid_labels_ets.end(), 0);
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
  ParallelPreflowPushAlgorithm(const PreflowPushContext &ctx);
  ~ParallelPreflowPushAlgorithm() override = default;

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

  void discharge_active_nodes();

  void apply_updates();

  void discharge(const NodeID u);

  void atomic_discharge(
      NodeID u, BufferedVector<NodeID>::Buffer next_active_nodes, std::size_t &local_work_amount
  );

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

  std::size_t _round;

  bool _force_global_relabel;
  tbb::enumerable_thread_specific<std::size_t> _work_ets;
  GlobalRelabelingThreshold _grt;
  ParallelBFSRunner _parallel_bfs_runner;

  ScalableVector<NodeID> _nodes_to_desaturate;

  StaticArray<NodeID> _last_activated;
  StaticArray<NodeID> _cur_edge_offsets;

  StaticArray<EdgeWeight> _excess;
  StaticArray<EdgeWeight> _excess_delta;

  StaticArray<NodeID> _heights;
  StaticArray<NodeID> _next_heights;

  BufferedVector<NodeID> _active_nodes;
  BufferedVector<NodeID> _next_active_nodes;
  tbb::concurrent_vector<NodeID> _acitve_sink_nodes;
};

} // namespace kaminpar::shm
