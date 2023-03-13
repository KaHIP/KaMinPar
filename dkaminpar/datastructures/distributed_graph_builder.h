/*******************************************************************************
 * @file:   distributed_graph_builder.h
 * @author: Daniel Seemaier
 * @date:   27.10.2021
 * @brief:  Utility class to build a distributed graph from an edge list.
 ******************************************************************************/
#pragma once

#include <unordered_map>

#include <tbb/concurrent_hash_map.h>

#include "growt.h"

#include "dkaminpar/definitions.h"
#include "dkaminpar/graphutils/communication.h"

#include "common/parallel/atomic.h"
#include "common/scalable_vector.h"

namespace kaminpar::dist::graph {
class GhostNodeMapper {
  using GhostNodeMap = tbb::concurrent_hash_map<GlobalNodeID, NodeID>;

public:
  struct Result {
    growt::StaticGhostNodeMapping global_to_ghost;
    scalable_vector<GlobalNodeID> ghost_to_global;
    scalable_vector<PEID> ghost_owner;
  };

  explicit GhostNodeMapper(
      MPI_Comm comm, const scalable_vector<GlobalNodeID> &node_distribution)
      : _node_distribution{node_distribution} {
    const PEID rank = mpi::get_comm_rank(comm);
    _n = static_cast<NodeID>(_node_distribution[rank + 1] -
                             _node_distribution[rank]);
    _next_ghost_node = _n;
  }

  explicit GhostNodeMapper(
      const scalable_vector<GlobalNodeID> &node_distribution, const PEID rank)
      : _node_distribution{node_distribution} {
    _n = static_cast<NodeID>(_node_distribution[rank + 1] -
                             _node_distribution[rank]);
    _next_ghost_node = _n;
  }

  NodeID new_ghost_node(const GlobalNodeID global_node) {
    GhostNodeMap::accessor entry;
    if (_global_to_ghost.insert(entry, global_node)) {
      const NodeID ghost_node =
          _next_ghost_node.fetch_add(1, std::memory_order_relaxed);
      entry->second = ghost_node;
    } else {
      [[maybe_unused]] const bool found =
          _global_to_ghost.find(entry, global_node);
      KASSERT(found);
    }

    return entry->second;
  }

  [[nodiscard]] Result finalize() {
    const auto ghost_n = static_cast<NodeID>(_next_ghost_node - _n);

    growt::StaticGhostNodeMapping global_to_ghost(ghost_n);
    scalable_vector<GlobalNodeID> ghost_to_global(ghost_n);
    scalable_vector<PEID> ghost_owner(ghost_n);

    tbb::parallel_for(_global_to_ghost.range(), [&](const auto r) {
      for (auto it = r.begin(); it != r.end(); ++it) {
        const GlobalNodeID global_node = it->first;
        const NodeID local_node = it->second;
        const NodeID local_ghost = local_node - _n;
        const auto owner_it =
            std::upper_bound(_node_distribution.begin() + 1,
                             _node_distribution.end(), global_node);
        const PEID owner = static_cast<PEID>(
            std::distance(_node_distribution.begin(), owner_it) - 1);

        ghost_to_global[local_ghost] = global_node;
        ghost_owner[local_ghost] = owner;
        global_to_ghost.insert(global_node + 1,
                               local_node); // 0 cannot be used as a key
      }
    });

    return {.global_to_ghost = std::move(global_to_ghost),
            .ghost_to_global = std::move(ghost_to_global),
            .ghost_owner = std::move(ghost_owner)};
  }

private:
  scalable_vector<GlobalNodeID> _node_distribution;
  NodeID _n;
  parallel::Atomic<NodeID> _next_ghost_node;
  GhostNodeMap _global_to_ghost;
};

class Builder {
  SET_DEBUG(false);

public:
  Builder(MPI_Comm const comm) : _comm{comm} {}

  Builder &initialize(const NodeID n) {
    return initialize(
        mpi::build_distribution_from_local_count<GlobalNodeID, scalable_vector>(
            n, _comm));
  }

  Builder &initialize(scalable_vector<GlobalNodeID> node_distribution) {
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

  Builder &change_local_node_weight(const NodeID node,
                                    const NodeWeight weight) {
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
    NodeID local_v = is_local_node(global_v) ? global_v - _offset_n
                                             : create_ghost_node(global_v);
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
    auto edge_distribution =
        mpi::build_distribution_from_local_count<GlobalEdgeID, scalable_vector>(
            m, _comm);

    DistributedGraph graph{std::move(_node_distribution),
                           std::move(edge_distribution),
                           std::move(_nodes),
                           std::move(_edges),
                           std::move(_node_weights),
                           std::move(_edge_weights),
                           std::move(_ghost_owner),
                           std::move(_ghost_to_global),
                           std::move(_global_to_ghost),
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
            tbb::parallel_for<std::size_t>(
                0, buffer.size(), [&](const std::size_t i) {
                  const auto &[local_node_on_other_pe, weight] = buffer[i];
                  const NodeID local_node = graph.global_to_local_node(
                      graph.offset_n(pe) + local_node_on_other_pe);
                  graph.set_ghost_node_weight(local_node, weight);
                });
          });
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
    auto it = std::upper_bound(_node_distribution.begin() + 1,
                               _node_distribution.end(), global_u);
    KASSERT(it != _node_distribution.end());
    return static_cast<PEID>(std::distance(_node_distribution.begin(), it) - 1);
  }

  MPI_Comm _comm;

  scalable_vector<GlobalNodeID> _node_distribution;
  GlobalNodeID _offset_n{0};
  NodeID _local_n{0};

  scalable_vector<EdgeID> _nodes{};
  scalable_vector<NodeID> _edges{};
  scalable_vector<NodeWeight> _node_weights{};
  scalable_vector<EdgeWeight> _edge_weights{};
  scalable_vector<PEID> _ghost_owner{};
  scalable_vector<GlobalNodeID> _ghost_to_global{};
  std::unordered_map<GlobalNodeID, NodeID> _global_to_ghost{};

  bool _unit_node_weights{true};
};
} // namespace kaminpar::dist::graph
