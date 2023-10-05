/*******************************************************************************
 * @file:   distributed_graph_builder.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Utility class to build a distributed graph from an edge list.
 ******************************************************************************/
#pragma once

#include <unordered_map>

#include <tbb/concurrent_hash_map.h>

#include "kaminpar-dist/datastructures/ghost_node_mapper.h"
#include "kaminpar-dist/datastructures/growt.h"
#include "kaminpar-dist/dkaminpar.h"
#include "kaminpar-dist/graphutils/communication.h"

#include "kaminpar-common/datastructures/scalable_vector.h"
#include "kaminpar-common/parallel/atomic.h"

namespace kaminpar::dist::graph {
[[nodiscard]] inline growt::StaticGhostNodeMapping
build_static_ghost_node_mapping(std::unordered_map<GlobalNodeID, NodeID> global_to_ghost) {
  growt::StaticGhostNodeMapping static_mapping(global_to_ghost.size());
  for (const auto &[key, value] : global_to_ghost) {
    static_mapping.insert(key + 1, value); // 0 cannot be used as a key in growt hash tables
  }
  return static_mapping;
}

class Builder {
  SET_DEBUG(false);

public:
  Builder(MPI_Comm const comm) : _comm{comm} {}

  template <typename T> using vec = std::vector<T>;
  Builder &initialize(const NodeID n) {
    return initialize(mpi::build_distribution_from_local_count<GlobalNodeID, vec>(n, _comm));
  }

  Builder &initialize(std::vector<GlobalNodeID> node_distribution) {
    _node_distribution = std::move(node_distribution);

    const int rank = mpi::get_comm_rank(_comm);
    _offset_n = _node_distribution[rank];
    _local_n = _node_distribution[rank + 1] - _node_distribution[rank];

    return *this;
  }

  Builder &create_node(const NodeWeight weight) {
    _nodes.push_back(_edges.size());
    _node_weights.push_back(weight);
    _unit_node_weights = _unit_node_weights && (weight == 1);

    return *this;
  }

  Builder &change_local_node_weight(const NodeID node, const NodeWeight weight) {
    KASSERT(node < _node_weights.size());
    _node_weights[node] = weight;
    _unit_node_weights = _unit_node_weights && (weight == 1);

    return *this;
  }

  Builder &add_local_node_weight(const NodeID node, const NodeWeight delta) {
    KASSERT(node < _node_weights.size());
    _node_weights[node] += delta;
    _unit_node_weights = _unit_node_weights && (delta == 0);

    return *this;
  }

  Builder &create_edge(const EdgeWeight weight, const GlobalNodeID global_v) {
    NodeID local_v = is_local_node(global_v) ? global_v - _offset_n : create_ghost_node(global_v);
    _edges.push_back(local_v);
    _edge_weights.push_back(weight);

    return *this;
  }

  DistributedGraph finalize() {
    _nodes.push_back(_edges.size());

    // First step: use unit node weights for ghost nodes
    // If needed, we will update those afterwards once we have the graph data
    // structure
    for (NodeID ghost_u = 0; ghost_u < _ghost_to_global.size(); ++ghost_u) {
      _node_weights.push_back(1);
    }

    const EdgeID m = _edges.size();
    auto edge_distribution = mpi::build_distribution_from_local_count<GlobalEdgeID, vec>(m, _comm);

    DistributedGraph graph{
        static_array::create_from(_node_distribution),
        static_array::create_from(edge_distribution),
        static_array::create_from(_nodes),
        static_array::create_from(_edges),
        static_array::create_from(_node_weights),
        static_array::create_from(_edge_weights),
        static_array::create_from(_ghost_owner),
        static_array::create_from(_ghost_to_global),
        build_static_ghost_node_mapping(_global_to_ghost),
        false,
        _comm};

    // If the graph does not have unit node weights, exchange ghost node weights
    // now
    struct Message {
      NodeID node;
      NodeWeight weight;
    };

    if (!_unit_node_weights) {
      mpi::graph::sparse_alltoall_interface_to_pe<Message>(
          graph,
          [&](const NodeID u) -> Message {
            return {u, graph.node_weight(u)};
          },
          [&](const auto buffer, const PEID pe) {
            tbb::parallel_for<std::size_t>(0, buffer.size(), [&](const std::size_t i) {
              const auto &[local_node_on_other_pe, weight] = buffer[i];
              const NodeID local_node =
                  graph.global_to_local_node(graph.offset_n(pe) + local_node_on_other_pe);
              graph.set_ghost_node_weight(local_node, weight);
            });
          }
      );
    }

    return graph;
  }

private:
  [[nodiscard]] bool is_local_node(const GlobalNodeID global_u) const {
    return _offset_n <= global_u && global_u < _offset_n + _local_n;
  }

  NodeID create_ghost_node(const GlobalNodeID global_u) {
    if (_global_to_ghost.find(global_u) == _global_to_ghost.end()) {
      const NodeID local_id = _local_n + _ghost_to_global.size();
      _ghost_to_global.push_back(global_u);
      _global_to_ghost[global_u] = local_id;
      _ghost_owner.push_back(find_ghost_owner(global_u));
    }

    return _global_to_ghost[global_u];
  }

  PEID find_ghost_owner(const GlobalNodeID global_u) const {
    auto it = std::upper_bound(_node_distribution.begin() + 1, _node_distribution.end(), global_u);
    KASSERT(it != _node_distribution.end());
    return static_cast<PEID>(std::distance(_node_distribution.begin(), it) - 1);
  }

  MPI_Comm _comm;

  std::vector<GlobalNodeID> _node_distribution;
  GlobalNodeID _offset_n{0};
  NodeID _local_n{0};

  std::vector<EdgeID> _nodes{};
  std::vector<NodeID> _edges{};
  std::vector<NodeWeight> _node_weights{};
  std::vector<EdgeWeight> _edge_weights{};
  std::vector<PEID> _ghost_owner{};
  std::vector<GlobalNodeID> _ghost_to_global{};
  std::unordered_map<GlobalNodeID, NodeID> _global_to_ghost{};

  bool _unit_node_weights{true};
};
} // namespace kaminpar::dist::graph
