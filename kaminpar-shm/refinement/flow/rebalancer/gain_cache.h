#pragma once

#include <algorithm>
#include <cstddef>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

template <typename PartitionedGraph, typename Graph> class NonConcurrentDenseGainCache {
public:
  void initialize(const PartitionedGraph &p_graph, const Graph &graph) {
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
      _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
        const BlockID v_block = _p_graph->block(v);
        _gain_cache[index(u, v_block)] += w;
      });
    }
  }

  void move(const NodeID u, const BlockID from, const BlockID to) {
    _graph->adjacent_nodes(u, [&](const NodeID v, const EdgeWeight w) {
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
  const PartitionedGraph *_p_graph;
  const Graph *_graph;

  std::size_t _n;
  std::size_t _k;

  StaticArray<EdgeWeight> _gain_cache;
};

} // namespace kaminpar::shm
