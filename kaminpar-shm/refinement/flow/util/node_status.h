#pragma once

#include <algorithm>
#include <cstdint>
#include <span>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

class NodeStatus {
public:
  static constexpr std::uint8_t kUnknown = 0;
  static constexpr std::uint8_t kSource = 1;
  static constexpr std::uint8_t kSink = 2;

  void initialize(const NodeID num_nodes) {
    _num_nodes = num_nodes;

    if (_node_status.size() < num_nodes) {
      _node_status.resize(num_nodes, static_array::noinit);
    }

    reset();
  }

  void reset() {
    std::fill_n(_node_status.begin(), _num_nodes, kUnknown);
    _source_nodes.clear();
    _sink_nodes.clear();
  }

  void add_source(const NodeID u) {
    KASSERT(is_unknown(u));

    _node_status[u] = kSource;
    _source_nodes.push_back(u);
  }

  void add_sink(const NodeID u) {
    KASSERT(is_unknown(u));

    _node_status[u] = kSink;
    _sink_nodes.push_back(u);
  }

  bool has_status(const NodeID u, const std::uint8_t status) const {
    KASSERT(u < _num_nodes);
    return _node_status[u] == status;
  }

  bool is_unknown(const NodeID u) const {
    return has_status(u, kUnknown);
  }

  bool is_terminal(const NodeID u) const {
    return !is_unknown(u);
  }

  bool is_source(const NodeID u) const {
    return has_status(u, kSource);
  }

  bool is_sink(const NodeID u) const {
    return has_status(u, kSink);
  }

  std::span<const NodeID> source_nodes() const {
    return _source_nodes;
  }

  std::span<const NodeID> sink_nodes() const {
    return _sink_nodes;
  }

private:
  NodeID _num_nodes;
  StaticArray<std::uint8_t> _node_status;

  ScalableVector<NodeID> _source_nodes;
  ScalableVector<NodeID> _sink_nodes;
};

} // namespace kaminpar::shm
