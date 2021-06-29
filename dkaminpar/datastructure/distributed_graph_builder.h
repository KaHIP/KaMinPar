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
  void initialize(const DNodeID global_n, const DEdgeID global_m, const PEID rank,
                  scalable_vector<DNodeID> node_distribution) {
    ASSERT(static_cast<std::size_t>(rank + 1) < node_distribution.size());
    ASSERT(global_n == node_distribution.back());
    ASSERT(0 == node_distribution.front());

    _global_n = global_n;
    _global_m = global_m;
    _node_distribution = std::move(node_distribution);
    _offset_n = _node_distribution[rank];
    _local_n = _node_distribution[rank + 1] - _node_distribution[rank];
  }

  void create_node(const DNodeWeight weight) {
    _nodes.push_back(_edges.size());
    _node_weights.push_back(weight);
  }

  void create_edge(const DEdgeWeight weight, const DNodeID global_v) {
    DNodeID local_v = is_local_node(global_v) ? global_v - _offset_n : create_ghost_node(global_v);
    _edges.push_back(local_v);
    _edge_weights.push_back(weight);
  }

  DistributedGraph finalize() {
    _nodes.push_back(_edges.size());
    for (DNodeID ghost_u = 0; ghost_u < _ghost_to_global.size(); ++ghost_u) {
      _node_weights.push_back(1); // TODO support weighted instances
    }

    // build edge distribution array
    const DEdgeID m = _edges.size();
    DEdgeID offset_m = 0;
    MPI_Exscan(&m, &offset_m, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

    const auto [size, rank] = mpi::get_comm_info();
    scalable_vector<DEdgeID> edge_distribution(size + 1);
    MPI_Allgather(&offset_m, 1, MPI_UINT64_T, edge_distribution.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);
    edge_distribution.back() = _global_m;

    DBG << "Finalized graph: " << V(offset_m) << V(edge_distribution);

    return {_global_n,
            _global_m,
            _ghost_to_global.size(),
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
  [[nodiscard]] bool is_local_node(const DNodeID global_u) const {
    return _offset_n <= global_u && global_u < _offset_n + _local_n;
  }

  DNodeID create_ghost_node(const DNodeID global_u) {
    if (!_global_to_ghost.contains(global_u)) {
      const DNodeID local_id = _local_n + _ghost_to_global.size();
      _ghost_to_global.push_back(global_u);
      _global_to_ghost[global_u] = local_id;
      _ghost_owner.push_back(find_ghost_owner(global_u));
    }

    return _global_to_ghost[global_u];
  }

  PEID find_ghost_owner(const DNodeID global_u) const {
    auto it = std::upper_bound(_node_distribution.begin() + 1, _node_distribution.end(), global_u);
    ASSERT(it != _node_distribution.end());
    return static_cast<PEID>(std::distance(_node_distribution.begin(), it) - 1);
  }

  DNodeID _global_n;
  DEdgeID _global_m;

  scalable_vector<DNodeID> _node_distribution;
  DNodeID _offset_n{0};
  DNodeID _local_n{0};

  scalable_vector<DEdgeID> _nodes{};
  scalable_vector<DNodeID> _edges{};
  scalable_vector<DNodeWeight> _node_weights{};
  scalable_vector<DEdgeWeight> _edge_weights{};
  scalable_vector<PEID> _ghost_owner{};
  scalable_vector<DNodeID> _ghost_to_global{};
  std::unordered_map<DNodeID, DNodeID> _global_to_ghost{};
};
} // namespace dkaminpar::graph