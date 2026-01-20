#pragma once

#include "kaminpar-shm/coarsening/sparsification/sampler.h"
#include "kaminpar-shm/datastructures/csr_graph.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::sparsification {

class UniformRandomSampler : public Sampler {
public:
  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override;
};

class WeightedUniformRandomSampler : public Sampler {
public:
  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID desired_num_edges) override;
};

} // namespace kaminpar::shm::sparsification
