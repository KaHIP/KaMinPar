/*******************************************************************************
 * @file:   border_tracker.h
 * @author: Daniel Seemaier
 * @date:   14.03.2023
 ******************************************************************************/
#pragma once

#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
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
    for (auto &nodes : _local_border_nodes) {
      nodes.clear();
    }

    tbb::parallel_for(
        tbb::blocked_range<NodeID>(0, p_graph.graph().n()),
        [&](const tbb::blocked_range<NodeID> &range) {
          // Resolve thread-local storage once per range, rather than once per border node.
          auto &nodes = _local_border_nodes.local();
          for (NodeID u = range.begin(); u < range.end(); ++u) {
            if (_gain_cache.is_border_node(u, p_graph.block(u))) {
              nodes.push_back(u);
            }
            _node_tracker.set(u, 0);
          }
        }
    );

    assign_from_thread_locals(_local_border_nodes);
    _next_border_node = 0;
  }

  template <typename Container>
  void init_precomputed(const PartitionedGraph &p_graph, const Container &border_nodes) {
    _border_nodes.assign(border_nodes.begin(), border_nodes.end());
    tbb::parallel_for<NodeID>(0, p_graph.graph().n(), [&](const NodeID u) {
      _node_tracker.set(u, 0);
    });
    _next_border_node = 0;
  }

  template <typename Lambda> NodeID poll(const NodeID count, int id, Lambda &&lambda) {
    NodeID polled = 0;
    const NodeID *nodes = _border_nodes.data();
    const std::size_t size = _border_nodes.size();

    while (polled < count && _next_border_node < size) {
      const NodeID remaining = count - polled;
      const NodeID from = _next_border_node.fetch_add(remaining);
      const NodeID to = std::min<NodeID>(from + remaining, size);

      for (NodeID current = from; current < to; ++current) {
        const NodeID node = nodes[current];
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
  void assign_from_thread_locals(
      const tbb::enumerable_thread_specific<std::vector<NodeID>> &local_border_nodes
  ) {
    std::vector<const std::vector<NodeID> *> local_vectors;
    std::vector<std::size_t> offsets;

    std::size_t total_size = 0;
    for (const std::vector<NodeID> &local_nodes : local_border_nodes) {
      local_vectors.push_back(&local_nodes);
      offsets.push_back(total_size);
      total_size += local_nodes.size();
    }

    _border_nodes.resize(total_size);
    tbb::parallel_for<std::size_t>(0, local_vectors.size(), [&](const std::size_t i) {
      std::copy(
          local_vectors[i]->begin(), local_vectors[i]->end(), _border_nodes.begin() + offsets[i]
      );
    });
  }

  GainCache &_gain_cache;
  NodeTracker &_node_tracker;

  std::vector<NodeID> _border_nodes;
  tbb::enumerable_thread_specific<std::vector<NodeID>> _local_border_nodes;

  // Pollers write this counter concurrently while reading the immutable vector metadata.
  alignas(64) parallel::Atomic<NodeID> _next_border_node;
};

} // namespace kaminpar::shm::fm
