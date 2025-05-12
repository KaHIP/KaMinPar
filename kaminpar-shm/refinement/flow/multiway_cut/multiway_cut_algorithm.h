#pragma once

#include <unordered_set>
#include <vector>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/datastructures/partitioned_graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {

class MultiwayCutAlgorithm {
public:
  struct Result {
    EdgeWeight cut_value;
    std::unordered_set<EdgeID> cut_edges;
  };

  virtual ~MultiwayCutAlgorithm() = default;

  [[nodiscard]] virtual Result
  compute(const CSRGraph &graph, const std::vector<std::unordered_set<NodeID>> &terminal_sets) = 0;

  [[nodiscard]] virtual Result compute(
      const PartitionedCSRGraph &p_graph,
      const CSRGraph &graph,
      const std::vector<std::unordered_set<NodeID>> &terminal_sets
  );
};

namespace debug {

[[nodiscard]] bool is_valid_multiway_cut(
    const CSRGraph &graph,
    const std::vector<std::unordered_set<NodeID>> &terminal_sets,
    const std::unordered_set<EdgeID> &cut_edges
);

}

} // namespace kaminpar::shm
