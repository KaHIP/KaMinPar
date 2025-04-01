//
// Created by badger on 5/6/24.
//

#ifndef SAMPLER_H
#define SAMPLER_H
#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

#include "kaminpar-common/datastructures/static_array.h"

namespace kaminpar::shm::sparsification {

class Sampler {
public:
  virtual StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) = 0;
  virtual ~Sampler() = default;
};

} // namespace kaminpar::shm::sparsification

#endif // SAMPLER_H
