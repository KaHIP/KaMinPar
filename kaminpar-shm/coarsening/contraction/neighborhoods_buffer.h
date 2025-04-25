/*******************************************************************************
 * @file:   neighborhoods_buffer.h
 * @author: Daniel Salwasser
 * @date:   12.04.2024
 ******************************************************************************/
#pragma once

#include <array>
#include <cstring>

#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/compact_static_array.h"

namespace kaminpar::shm::contraction {

class NeighborhoodsBuffer {
  static constexpr NodeID kMaxNumNodes = 4096;
  static constexpr EdgeID kMaxNumEdges = 32768;

public:
  [[nodiscard]] static bool exceeds_capacity(const NodeID degree) {
    return degree >= kMaxNumEdges;
  }

  NeighborhoodsBuffer(
      EdgeID *nodes,
      NodeID *edges,
      NodeWeight *node_weights,
      EdgeWeight *edge_weights,
      CompactStaticArray<NodeID> &remapping
  )
      : _nodes(nodes),
        _edges(edges),
        _node_weights(node_weights),
        _edge_weights(edge_weights),
        _remapping(remapping),
        _num_buffered_nodes(0),
        _num_buffered_edges(0) {
    _node_buffer[0] = 0;
  }

  [[nodiscard]] NodeID num_buffered_nodes() const {
    return _num_buffered_nodes;
  }

  [[nodiscard]] NodeID num_buffered_edges() const {
    return _num_buffered_edges;
  }

  [[nodiscard]] bool overfills(const NodeID degree) const {
    return _num_buffered_nodes + 1 >= kMaxNumNodes || _num_buffered_edges + degree >= kMaxNumEdges;
  }

  template <typename Lambda>
  void add(const NodeID c_u, const NodeID degree, const NodeWeight weight, Lambda &&it) {
    _remapping_buffer[_num_buffered_nodes] = c_u;
    _node_buffer[_num_buffered_nodes + 1] = degree + _node_buffer[_num_buffered_nodes];
    _node_weight_buffer[_num_buffered_nodes] = weight;
    _num_buffered_nodes += 1;

    it([this](const auto c_v, const auto weight) {
      _edge_buffer[_num_buffered_edges] = c_v;
      _edge_weight_buffer[_num_buffered_edges] = weight;
      _num_buffered_edges += 1;
    });
  }

  void flush(const NodeID c_u, const EdgeID e) {
    const std::size_t buffered_edge_weights_size = _num_buffered_edges * sizeof(EdgeWeight);
    std::memcpy(_edge_weights + e, _edge_weight_buffer.data(), buffered_edge_weights_size);

    const std::size_t buffered_edges_size = _num_buffered_edges * sizeof(NodeID);
    std::memcpy(_edges + e, _edge_buffer.data(), buffered_edges_size);

    const std::size_t buffered_node_weights_size = _num_buffered_nodes * sizeof(NodeWeight);
    std::memcpy(_node_weights + c_u, _node_weight_buffer.data(), buffered_node_weights_size);

    for (NodeID i = 0; i < _num_buffered_nodes; ++i) {
      _remapping.write(_remapping_buffer[i], c_u + i);
      _nodes[c_u + i] = _node_buffer[i] + e;
    }

    _num_buffered_nodes = 0;
    _num_buffered_edges = 0;
    _node_buffer[0] = 0;
  }

private:
  EdgeID *_nodes;
  NodeID *_edges;
  NodeWeight *_node_weights;
  EdgeWeight *_edge_weights;
  CompactStaticArray<NodeID> &_remapping;

  NodeID _num_buffered_nodes;
  EdgeID _num_buffered_edges;
  std::array<NodeID, kMaxNumNodes> _remapping_buffer;
  std::array<EdgeID, kMaxNumNodes> _node_buffer;
  std::array<NodeWeight, kMaxNumNodes> _node_weight_buffer;
  std::array<NodeID, kMaxNumEdges> _edge_buffer;
  std::array<EdgeWeight, kMaxNumEdges> _edge_weight_buffer;
};

} // namespace kaminpar::shm::contraction
