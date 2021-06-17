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

#include "distributed_definitions.h"

#include <definitions.h>
#include <ranges>
#include <vector>

namespace dkaminpar {
class DistributedGraph {
public:
  // Graph size
  [[nodiscard]] inline DNodeID global_n() const { return _global_n; }
  [[nodiscard]] inline DEdgeID global_m() const { return _global_m; }
  [[nodiscard]] inline DNodeID local_n() const { return _local_n; }
  [[nodiscard]] inline DNodeID local_offset() const { return _local_offset; }
  [[nodiscard]] inline DNodeID n() const { return _nodes.size() - 1; }
  [[nodiscard]] inline DEdgeID m() const { return _edges.size(); }

  // Node type
  [[nodiscard]] inline bool is_ghost_node(const DNodeID u) const {
    ASSERT(u < n());
    return u >= local_n();
  }
  [[nodiscard]] inline bool is_owned_node(const DNodeID u) const {
    ASSERT(u < n());
    return u < local_n();
  }

  // Distributed info
  [[nodiscard]] inline PEID ghost_owner(const DNodeID u) const {
    ASSERT(u < n());
    ASSERT(is_ghost_node(u));
    return _ghost_owner[u - local_n()];
  }

  [[nodiscard]] inline DNodeID global_node(const DNodeID local_u) const {
    ASSERT(local_u < n());

    if (is_owned_node(local_u)) {
      return _local_offset + local_u;
    } else {
      return _ghost_to_global[local_u - local_n()];
    }
  }

  [[nodiscard]] inline DNodeID local_node(const DNodeID global_u) const {
    if (local_offset() <= global_u && global_u < local_offset() + local_n()) {
      return global_u - local_offset();
    } else {
      return 0;
    }
  }

  // Access methods
  [[nodiscard]] inline DNodeWeight node_weight(const DNodeID u) const {
    ASSERT(u < n());
    ASSERT(u < _node_weights.size());
    return _node_weights[u];
  }

  [[nodiscard]] inline DEdgeWeight edge_weight(const DEdgeID e) const {
    ASSERT(e < m());
    ASSERT(e < _edge_weights.size());
    return _edge_weights[e];
  }

  // Graph structure
  [[nodiscard]] inline DEdgeID first_edge(const DNodeID u) const {
    ASSERT(u < n());
    return _nodes[u];
  }

  [[nodiscard]] inline DEdgeID first_invalid_edge(const DNodeID u) const {
    ASSERT(u < n());
    return _nodes[u + 1];
  }

  [[nodiscard]] inline DNodeID edge_target(const DEdgeID e) const {
    ASSERT(e < m());
    return _edges[e];
  }

private:
  DNodeID _global_n;
  DEdgeID _global_m;
  DNodeID _local_n;
  DNodeID _local_offset;

  std::vector<DEdgeID> _nodes;
  std::vector<DNodeID> _edges;
  std::vector<DNodeWeight> _node_weights;
  std::vector<DEdgeWeight> _edge_weights;

  std::vector<PEID> _ghost_owner;
  std::vector<DNodeID> _ghost_to_global;
  // _global_to_ghost
};
} // namespace dkaminpar