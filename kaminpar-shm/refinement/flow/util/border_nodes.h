#pragma once

#include <oneapi/tbb/parallel_for.h>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"

#include "kaminpar-common/datastructures/static_array.h"
#include "kaminpar-common/timer.h"

namespace kaminpar::shm {

class BorderNodes {
public:
  void initialize(const PartitionedGraph &p_graph, const CSRGraph &graph) {
    SCOPED_TIMER("Compute border nodes");

    const NodeID num_nodes = graph.n();
    if (_flags.size() < num_nodes) {
      _flags.resize(num_nodes, static_array::noinit);
    }

    tbb::parallel_for<NodeID>(0, num_nodes, [&](const NodeID u) {
      const BlockID u_block = p_graph.block(u);

      bool is_border_node = false;
      graph.adjacent_nodes(u, [&](const NodeID v) {
        const BlockID v_block = p_graph.block(v);
        if (u_block == v_block) {
          return false;
        }

        is_border_node = true;
        return true;
      });

      _flags[u] = is_border_node;
    });
  }

  [[nodiscard]] bool is_border_node(const NodeID u) const {
    return _flags[u];
  }

private:
  StaticArray<bool> _flags;
};

} // namespace kaminpar::shm
