#pragma once

#include <algorithm>
#include <span>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm {

class BorderRegion {
public:
  BorderRegion() : _block(kInvalidBlockID), _max_weight(0), _cur_weight(0) {};

  BorderRegion(const BorderRegion &) = delete;
  BorderRegion &operator=(const BorderRegion &) = delete;

  BorderRegion(BorderRegion &&) noexcept = default;
  BorderRegion &operator=(BorderRegion &&) noexcept = default;

  void reset(const BlockID block, const NodeWeight max_weight, const NodeID max_num_nodes) {
    _block = block;
    _max_weight = max_weight;

    _cur_weight = 0;

    if (_node_status.size() < max_num_nodes) {
      _node_status.resize(max_num_nodes, static_array::noinit);
      std::fill_n(_node_status.data(), max_num_nodes, 0);
    } else {
      for (const NodeID u : _nodes) {
        _node_status[u] = 0;
      }

      _nodes.clear();
    }
  }

  void insert(NodeID u, NodeWeight u_weight) {
    KASSERT(!contains(u));
    KASSERT(fits(u_weight));

    _node_status[u] = 1;
    _nodes.push_back(u);
    _cur_weight += u_weight;
  }

  [[nodiscard]] bool fits(NodeWeight weight) const {
    return _cur_weight + weight <= _max_weight;
  }

  [[nodiscard]] bool contains(NodeID u) const {
    KASSERT(u < _node_status.size());
    return _node_status[u] != 0;
  }

  [[nodiscard]] std::span<const NodeID> nodes() const {
    return _nodes;
  }

  [[nodiscard]] NodeID num_nodes() const {
    return _nodes.size();
  }

  [[nodiscard]] bool empty() const {
    return _nodes.size() == 0;
  }

  [[nodiscard]] NodeWeight weight() const {
    return _cur_weight;
  }

  [[nodiscard]] NodeWeight max_weight() const {
    return _max_weight;
  }

  [[nodiscard]] BlockID block() const {
    return _block;
  }

private:
  BlockID _block;
  NodeWeight _max_weight;

  NodeWeight _cur_weight;
  StaticArray<std::uint8_t> _node_status;
  ScalableVector<NodeID> _nodes;
};

} // namespace kaminpar::shm
