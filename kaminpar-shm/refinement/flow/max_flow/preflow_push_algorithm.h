#pragma once

#include <limits>
#include <span>
#include <unordered_set>
#include <vector>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/max_flow/max_flow_algorithm.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

class PreflowPushAlgorithm : public MaxFlowAlgorithm {
  SET_DEBUG(false);

  class Level {
  public:
    Level() : _num_nodes(0) {}

    bool emtpy() const {
      return _num_nodes == 0;
    }

    bool has_active_node() const {
      return !_active_nodes.empty();
    }

    NodeID pop_activate_node() {
      const NodeID u = _active_nodes.back();
      _active_nodes.pop_back();
      return u;
    }

    void activate(const NodeID u) {
      _active_nodes.push_back(u);
    }

    void add_active_node(const NodeID u) {
      _num_nodes += 1;
      activate(u);
    }

    void add_inactive_node([[maybe_unused]] const NodeID u) {
      _num_nodes += 1;
    }

    void remove([[maybe_unused]] const NodeID u) {
      _num_nodes -= 1;
    }

    void clear() {
      _num_nodes = 0;
      _active_nodes.clear();
    }

  private:
    NodeID _num_nodes;
    std::vector<NodeID> _active_nodes;
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
  PreflowPushAlgorithm(const PreflowPushContext &ctx);
  ~PreflowPushAlgorithm() override = default;

  void compute(
      const CSRGraph &graph,
      const std::unordered_set<NodeID> &sources,
      const std::unordered_set<NodeID> &sinks,
      std::span<EdgeWeight> flow
  ) override;

private:
  void create_reverse_edges_index();

  NodeID global_relabel();

  NodeID initialize_levels();

  void saturate_source_edges();

  void compute_exact_heights();

  NodeID discharge(NodeID u);

  bool push(NodeID from, NodeID to, EdgeID e, EdgeWeight residual_capacity);

  NodeID relabel(NodeID u);

private:
  const PreflowPushContext _ctx;

  const CSRGraph *_graph;
  const std::unordered_set<NodeID> *_sources;
  const std::unordered_set<NodeID> *_sinks;

  std::span<EdgeWeight> _flow;
  GlobalRelabelingThreshold _grt;

  StaticArray<EdgeID> _reverse_edges;
  StaticArray<NodeID> _heights;
  StaticArray<EdgeWeight> _excess;
  StaticArray<NodeID> _cur_edge_offsets;
  StaticArray<Level> _levels;
};

} // namespace kaminpar::shm
