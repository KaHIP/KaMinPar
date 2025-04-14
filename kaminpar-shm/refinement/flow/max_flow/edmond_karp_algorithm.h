#pragma once

#include <span>
#include <unordered_set>

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"
#include "kaminpar-shm/refinement/flow/max_flow/max_flow_algorithm.h"

#include "kaminpar-common/logger.h"

namespace kaminpar::shm {

class EdmondsKarpAlgorithm : public MaxFlowAlgorithm {
  SET_DEBUG(false);

public:
  EdmondsKarpAlgorithm() = default;
  ~EdmondsKarpAlgorithm() override = default;

  void compute(
      const CSRGraph &graph,
      const std::unordered_set<NodeID> &sources,
      const std::unordered_set<NodeID> &sinks,
      std::span<EdgeWeight> flow
  ) override;
};

} // namespace kaminpar::shm
