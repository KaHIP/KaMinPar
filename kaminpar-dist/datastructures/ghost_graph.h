/*******************************************************************************
 * Stores edges from ghost nodes to local nodes of the graph from which the ghost
 * graph is constructed.
 *
 * @file:   ghost_graph.h
 * @author: Daniel Seemaier
 * @date:   25.12.2024
 ******************************************************************************/
#pragma once

#include <tbb/parallel_for.h>

#include "kaminpar-dist/datastructures/abstract_distributed_graph.h"
#include "kaminpar-dist/dkaminpar.h"

#include "kaminpar-common/assert.h"
#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::dist {

class GhostGraph {
public:
  GhostGraph();

  GhostGraph(const DistributedGraph &graph);

  GhostGraph(const GhostGraph &) = delete;
  GhostGraph &operator=(const GhostGraph &) = delete;

  GhostGraph(GhostGraph &&) = default;
  GhostGraph &operator=(GhostGraph &&) = default;

  void initialize(const DistributedGraph &graph);

  [[nodiscard]] NodeID n() const {
    return static_cast<NodeID>(_xadj.size() - 1);
  }

  [[nodiscard]] EdgeID m() const {
    return static_cast<EdgeID>(_adjncy.size());
  }

  [[nodiscard]] bool is_edge_weighted() const {
    return !_adjwgt.empty();
  }

  template <typename Lambda> void pfor_nodes(Lambda &&lambda) {
    tbb::parallel_for<NodeID>(0, n(), [&](const NodeID u) { lambda(u); });
  }

  template <typename Lambda> void adjacent_nodes(const NodeID u, Lambda &&lambda) {
    KASSERT(u >= _n && u < _n + _ghost_n, "node " << u << " is not a ghost node");

    constexpr bool kDontDecodeEdgeWeights = std::is_invocable_v<Lambda, NodeID>;
    constexpr bool kDecodeEdgeWeights = std::is_invocable_v<Lambda, NodeID, EdgeWeight>;
    static_assert(kDontDecodeEdgeWeights || kDecodeEdgeWeights);

    for (EdgeID i = _xadj[u - _n]; i < _xadj[u - _n + 1]; ++i) {
      if constexpr (kDecodeEdgeWeights) {
        lambda(_adjncy[i], (is_edge_weighted() ? _adjwgt[i] : static_cast<EdgeWeight>(1)));
      } else {
        lambda(_adjncy[i]);
      }
    }
  }

private:
  NodeID _n = 0;
  NodeID _ghost_n = 0;

  StaticArray<EdgeID> _xadj;
  StaticArray<NodeID> _adjncy;
  StaticArray<EdgeWeight> _adjwgt;
};

} // namespace kaminpar::dist
