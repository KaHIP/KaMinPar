/*******************************************************************************
 * This file is part of KaMinPar.
 *
 * Copyright (C) 2021 Daniel Seemaier <daniel.seemaier@kit.edu>
 *
 * KaMinPar is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * KaMinPar is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with KaMinPar.  If not, see <http://www.gnu.org/licenses/>.
 *
******************************************************************************/
#pragma once

#include "dkaminpar/distributed_definitions.h"

#include <unordered_map>

namespace dkaminpar::graph {
class Builder {
  SET_DEBUG(false);

public:
  Builder &initialize(const GlobalNodeID global_n, const GlobalEdgeID global_m, const PEID rank,
                  scalable_vector<GlobalNodeID> node_distribution) {
    ASSERT(static_cast<std::size_t>(rank + 1) < node_distribution.size());
    ASSERT(global_n == node_distribution.back());
    ASSERT(0 == node_distribution.front());

    _global_n = global_n;
    _global_m = global_m;
    _node_distribution = std::move(node_distribution);
    _offset_n = _node_distribution[rank];
    _local_n = _node_distribution[rank + 1] - _node_distribution[rank];

    return *this;
  }

  Builder &create_node(const NodeWeight weight) {
    _nodes.push_back(_edges.size());
    _node_weights.push_back(weight);

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
    for (NodeID ghost_u = 0; ghost_u < _ghost_to_global.size(); ++ghost_u) {
      _node_weights.push_back(1); // TODO support weighted instances
    }

    // build edge distribution array
    GlobalEdgeID offset_m = mpi::exscan(static_cast<GlobalEdgeID>(_edges.size()), MPI_SUM, MPI_COMM_WORLD);

    const auto [size, rank] = mpi::get_comm_info();
    scalable_vector<GlobalEdgeID> edge_distribution(size + 1);
    mpi::allgather(&offset_m, 1, edge_distribution.data(), 1, MPI_COMM_WORLD);
    edge_distribution.back() = _global_m;

    DBG << "Finalized graph: " << V(offset_m) << V(edge_distribution);

    return {_global_n,
            _global_m,
            static_cast<NodeID>(_ghost_to_global.size()),
            _offset_n,
            offset_m,
            std::move(_node_distribution),
            std::move(edge_distribution),
            std::move(_nodes),
            std::move(_edges),
            std::move(_node_weights),
            std::move(_edge_weights),
            std::move(_ghost_owner),
            std::move(_ghost_to_global),
            std::move(_global_to_ghost)};
  }

private:
  [[nodiscard]] bool is_local_node(const GlobalNodeID global_u) const {
    return _offset_n <= global_u && global_u < _offset_n + _local_n;
  }

  NodeID create_ghost_node(const GlobalNodeID global_u) {
    if (!_global_to_ghost.contains(global_u)) {
      const NodeID local_id = _local_n + _ghost_to_global.size();
      _ghost_to_global.push_back(global_u);
      _global_to_ghost[global_u] = local_id;
      _ghost_owner.push_back(find_ghost_owner(global_u));
    }

    return _global_to_ghost[global_u];
  }

  PEID find_ghost_owner(const GlobalNodeID global_u) const {
    auto it = std::upper_bound(_node_distribution.begin() + 1, _node_distribution.end(), global_u);
    ASSERT(it != _node_distribution.end());
    return static_cast<PEID>(std::distance(_node_distribution.begin(), it) - 1);
  }

  GlobalNodeID _global_n;
  GlobalEdgeID _global_m;

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
};
} // namespace dkaminpar::graph