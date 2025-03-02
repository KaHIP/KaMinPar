/*******************************************************************************
 * @file:   border_tracker.h
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 ******************************************************************************/
#pragma once

#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>

#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/fm/node_tracker.h"

#include "kaminpar-common/parallel/atomic.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm::fm {

template <typename GainCache> class BorderNodes {
public:
  BorderNodes(GainCache &gain_cache, NodeTracker &node_tracker)
      : _gain_cache(gain_cache),
        _node_tracker(node_tracker) {}

  void init(const PartitionedGraph &p_graph) {
    _border_nodes.clear();
    tbb::parallel_for<NodeID>(0, p_graph.graph().n(), [&](const NodeID u) {
      if (_gain_cache.is_border_node(u, p_graph.block(u))) {
        _border_nodes.push_back(u);
      }
      _node_tracker.set(u, 0);
    });
    _next_border_node = 0;
  }

  template <typename Container>
  void init_precomputed(const PartitionedGraph &p_graph, const Container &border_nodes) {
    _border_nodes.clear();
    for (const auto &u : border_nodes) {
      _border_nodes.push_back(u);
    }
    tbb::parallel_for<NodeID>(0, p_graph.graph().n(), [&](const NodeID u) {
      _node_tracker.set(u, 0);
    });
    _next_border_node = 0;
  }

  template <typename Lambda> NodeID poll(const NodeID count, int id, Lambda &&lambda) {
    NodeID polled = 0;

    while (polled < count && _next_border_node < _border_nodes.size()) {
      const NodeID remaining = count - polled;
      const NodeID from = _next_border_node.fetch_add(remaining);
      const NodeID to = std::min<NodeID>(from + remaining, _border_nodes.size());

      for (NodeID current = from; current < to; ++current) {
        const NodeID node = _border_nodes[current];
        if (_node_tracker.owner(node) == NodeTracker::UNLOCKED && _node_tracker.lock(node, id)) {
          lambda(node);
          ++polled;
        }
      }
    }

    return polled;
  }

  [[nodiscard]] NodeID get() const {
    return has_more() ? _border_nodes[_next_border_node] : kInvalidNodeID;
  }

  [[nodiscard]] bool has_more() const {
    return _next_border_node < _border_nodes.size();
  }

  [[nodiscard]] std::size_t remaining() const {
    return _border_nodes.size() - std::min<std::size_t>(_border_nodes.size(), _next_border_node);
  }

  [[nodiscard]] std::size_t size() const {
    return _border_nodes.size();
  }

  void shuffle() {
    Random::instance().shuffle(_border_nodes.begin(), _border_nodes.end());
  }

private:
  GainCache &_gain_cache;
  NodeTracker &_node_tracker;

  parallel::Atomic<NodeID> _next_border_node;
  tbb::concurrent_vector<NodeID> _border_nodes;
};

} // namespace kaminpar::shm::fm
