#pragma once

#include <algorithm>
#include <cstddef>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

template <typename PartitionedGraph, typename Graph> class NonConcurrentDenseGainCache {
public:
  void
  initialize(const BlockID overloaded_block, const PartitionedGraph &p_graph, const Graph &graph) {
    _overloaded_block = overloaded_block;
    _p_graph = &p_graph;
    _graph = &graph;

    _n = graph.n();
    _k = p_graph.k();

    const std::size_t gain_cache_size = _n * _k;
    if (_gain_cache.size() < gain_cache_size) {
      _gain_cache.resize(gain_cache_size, static_array::noinit);
    }

    std::fill_n(_gain_cache.begin(), gain_cache_size, 0);
    for (const NodeID u : graph.nodes()) {
      if (_p_graph->block(u) != overloaded_block) {
        continue;
      }

      _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        const BlockID v_block = _p_graph->block(v);
        _gain_cache[index(u, v_block)] += w;
      });
    }
  }

  void move(const NodeID u, const BlockID from, const BlockID to) {
    _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      if (_p_graph->block(v) != _overloaded_block) {
        return;
      }

      _gain_cache[index(v, from)] -= w;
      _gain_cache[index(v, to)] += w;
    });
  }

  [[nodiscard]] EdgeWeight gain(const NodeID node, const BlockID from, const BlockID to) const {
    return connection(node, to) - connection(node, from);
  }

  [[nodiscard]] EdgeWeight connection(const NodeID node, const BlockID block) const {
    return _gain_cache[index(node, block)];
  }

private:
  [[nodiscard]] std::size_t index(const NodeID node, const BlockID block) const {
    return node * _k + block;
  }

private:
  BlockID _overloaded_block;

  const PartitionedGraph *_p_graph;
  const Graph *_graph;

  std::size_t _n;
  std::size_t _k;

  StaticArray<EdgeWeight> _gain_cache;
};

template <typename PartitionedGraph, typename Graph, typename PinnedNodeContainer>
class PinnedNonConcurrentDenseGainCache {
public:
  void initialize(
      const BlockID overloaded_block,
      const PinnedNodeContainer &pinned_nodes,
      const PartitionedGraph &p_graph,
      const Graph &graph
  ) {
    _overloaded_block = overloaded_block;
    _pinned_nodes = &pinned_nodes;

    _p_graph = &p_graph;
    _graph = &graph;

    _n = graph.n();
    _k = p_graph.k();

    const std::size_t gain_cache_size = _n * _k;
    if (_gain_cache.size() < gain_cache_size) {
      _gain_cache.resize(gain_cache_size, static_array::noinit);
    }

    std::fill_n(_gain_cache.begin(), gain_cache_size, 0);
    for (const NodeID u : graph.nodes()) {
      if (_p_graph->block(u) != overloaded_block && !_pinned_nodes->contains(u)) {
        continue;
      }

      _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        const BlockID v_block = _p_graph->block(v);
        _gain_cache[index(u, v_block)] += w;
      });
    }
  }

  void move(const NodeID u, const BlockID from, const BlockID to) {
    _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
      if (_p_graph->block(v) != _overloaded_block) {
        return;
      }

      _gain_cache[index(v, from)] -= w;
      _gain_cache[index(v, to)] += w;
    });
  }

  [[nodiscard]] EdgeWeight gain(const NodeID node, const BlockID from, const BlockID to) const {
    return connection(node, to) - connection(node, from);
  }

  [[nodiscard]] EdgeWeight connection(const NodeID node, const BlockID block) const {
    return _gain_cache[index(node, block)];
  }

private:
  [[nodiscard]] std::size_t index(const NodeID node, const BlockID block) const {
    return node * _k + block;
  }

private:
  BlockID _overloaded_block;
  const PinnedNodeContainer *_pinned_nodes;

  const PartitionedGraph *_p_graph;
  const Graph *_graph;

  std::size_t _n;
  std::size_t _k;

  StaticArray<EdgeWeight> _gain_cache;
};

} // namespace kaminpar::shm
