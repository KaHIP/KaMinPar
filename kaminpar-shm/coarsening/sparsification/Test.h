#pragma once

#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::sparsification {

class Sampler {
public:
  virtual ~Sampler() = default;

  virtual StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) = 0;
};

} // namespace kaminpar::shm::sparsification
