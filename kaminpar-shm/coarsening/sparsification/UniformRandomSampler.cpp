//
// Created by badger on 5/6/24.
//

#include "UniformRandomSampler.h"

#include "kaminpar-common/random.h"

namespace kaminpar::shm::sparsification {

StaticArray<EdgeWeight> UniformRandomSampler::sample(const CSRGraph &g) {
  unsigned int backedges_deleted = 0;
  unsigned int backedges = 0;
  unsigned int frontedges = 0;
  StaticArray<EdgeWeight> sample = StaticArray<EdgeWeight>(g.m());
  unsigned int edges_deleted = 0;
  for (int v = 0; v < g.n(); ++v) {
    for (EdgeID e : g.incident_edges(v)) {
      KASSERT(v != g.edge_target(e), "no loops allowed", assert::always);
      if (v < g.edge_target(e)) {
        frontedges++;
        sample[e] = Random::instance().random_bool(_probability) ? g.edge_weight(e) : 0;
        if (!sample[e])
          edges_deleted++;
      }
    }
  }
  return sample;
}
} // namespace kaminpar::shm::sparsification