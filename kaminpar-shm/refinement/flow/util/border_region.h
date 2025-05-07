#pragma once

#include <unordered_set>
#include <utility>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/assert.h"

namespace kaminpar::shm {

class BorderRegion {
public:
  BorderRegion() : _block(kInvalidBlockID), _max_weight(0), _cur_weight(0) {};

  BorderRegion(BlockID block, NodeWeight max_weight)
      : _block(block),
        _max_weight(max_weight),
        _cur_weight(0) {}

  BorderRegion(const BorderRegion &) = delete;
  BorderRegion &operator=(const BorderRegion &) = delete;

  BorderRegion(BorderRegion &&) noexcept = default;
  BorderRegion &operator=(BorderRegion &&) noexcept = default;

  void reset(BlockID block, NodeWeight max_weight) {
    _block = block;
    _max_weight = max_weight;

    _cur_weight = 0;
    _nodes.clear();
  }

  [[nodiscard]] BlockID block() const {
    return _block;
  }

  [[nodiscard]] NodeWeight max_weight() const {
    return _max_weight;
  }

  [[nodiscard]] NodeWeight weight() const {
    return _cur_weight;
  }

  [[nodiscard]] NodeID num_nodes() const {
    return _nodes.size();
  }

  [[nodiscard]] const std::unordered_set<NodeID> &nodes() const {
    return _nodes;
  }

  [[nodiscard]] bool contains(NodeID node) const {
    return _nodes.contains(node);
  }

  [[nodiscard]] bool fits(NodeWeight weight) const {
    return _cur_weight + weight <= _max_weight;
  }

  void insert(NodeID node, NodeWeight weight) {
    KASSERT(!contains(node), assert::always);
    KASSERT(fits(weight), assert::always);

    _nodes.insert(node);
    _cur_weight += weight;
  }

  void project(const std::unordered_map<NodeID, NodeID> &mapping) {
    std::unordered_set<NodeID> _new_nodes;

    for (const NodeID u : _nodes) {
      _new_nodes.insert(mapping.at(u));
    }

    std::swap(_nodes, _new_nodes);
  }

private:
  BlockID _block;
  NodeWeight _max_weight;

  NodeWeight _cur_weight;
  std::unordered_set<NodeID> _nodes;
};

} // namespace kaminpar::shm
