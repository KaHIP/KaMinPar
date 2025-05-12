#pragma once

#include <limits>
#include <unordered_set>

#include "kaminpar-shm/coarsening/contraction/cluster_contraction.h"
#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/max_flow/max_flow_algorithm.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

class HighestLevelPreflowPushAlgorithm : public MaxFlowAlgorithm {
  SET_DEBUG(false);

  class Level {
  public:
    bool emtpy() const {
      return _active_nodes.empty() && _inactive_nodes.empty();
    }

    bool has_active_node() const {
      return !_active_nodes.empty();
    }

    NodeID pop_activate_node() {
      KASSERT(has_active_node());

      auto it = _active_nodes.begin();
      const NodeID u = *it;

      _active_nodes.erase(it);
      _inactive_nodes.insert(u);

      return u;
    }

    void activate(const NodeID u) {
      KASSERT(_inactive_nodes.contains(u));

      _inactive_nodes.erase(u);
      _active_nodes.insert(u);
    }

    void add_active_node(const NodeID u) {
      KASSERT(!_active_nodes.contains(u));
      KASSERT(!_inactive_nodes.contains(u));

      _active_nodes.insert(u);
    }

    void add_inactive_node(const NodeID u) {
      KASSERT(!_active_nodes.contains(u));
      KASSERT(!_inactive_nodes.contains(u));

      _inactive_nodes.insert(u);
    }

    void remove_inactive_node(const NodeID u) {
      KASSERT(_inactive_nodes.contains(u));

      _inactive_nodes.erase(u);
    }

    [[nodiscard]] const std::unordered_set<NodeID> &active_nodes() const {
      return _active_nodes;
    }

    [[nodiscard]] const std::unordered_set<NodeID> &inactive_nodes() const {
      return _inactive_nodes;
    }

    void clear() {
      _active_nodes.clear();
      _inactive_nodes.clear();
    }

  private:
    std::unordered_set<NodeID> _active_nodes;
    std::unordered_set<NodeID> _inactive_nodes;
  };

  class GlobalRelabelingThreshold {
  public:
    GlobalRelabelingThreshold() : _threshold(std::numeric_limits<std::size_t>::max()), _work(0) {}

    GlobalRelabelingThreshold(const EdgeID num_edges, const double frequency)
        : _threshold(num_edges / frequency),
          _work(0) {}

    bool is_reached() const {
      return _work >= _threshold;
    }

    void add_work(std::size_t work) {
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
  HighestLevelPreflowPushAlgorithm(const HighestLevelPreflowPushContext &ctx);
  ~HighestLevelPreflowPushAlgorithm() override = default;

  void initialize(const CSRGraph &graph) override;

  void reset() override;

  Result compute_max_flow(
      const std::unordered_set<NodeID> &sources, const std::unordered_set<NodeID> &sinks
  ) override;

private:
  void saturate_source_edges();

  void find_maximum_preflow();

  void convert_maximum_preflow();

  void find_maximum_flow();

  template <bool kFirstPhase> NodeID global_relabel();

  void compute_exact_heights(const std::unordered_set<NodeID> &terminals);

  template <bool kFirstPhase> NodeID initialize_levels();

  void employ_gap_heuristic(NodeID height);

  template <bool kFirstPhase> NodeID discharge(NodeID u, NodeID u_height);

  bool push(NodeID from, NodeID to, EdgeID e, EdgeWeight residual_capacity);

  NodeID relabel(NodeID u, NodeID u_height);

private:
  const HighestLevelPreflowPushContext _ctx;

  const CSRGraph *_graph;
  StaticArray<EdgeID> _reverse_edge_index;

  const std::unordered_set<NodeID> *_sources;
  const std::unordered_set<NodeID> *_sinks;

  EdgeWeight _flow_value;
  StaticArray<EdgeWeight> _flow;

  GlobalRelabelingThreshold _grt;
  StaticArray<EdgeWeight> _excess;
  StaticArray<NodeID> _cur_edge_offsets;
  StaticArray<NodeID> _heights;
  ScalableVector<Level> _levels;
};

} // namespace kaminpar::shm
