//
// Created by badger on 5/6/24.
//

#include "UniformRandomSampler.h"

#include "kaminpar-common/random.h"

namespace kaminpar::shm::sparsification {

StaticArray<EdgeWeight> UniformRandomSampler::sample(const CSRGraph &g, EdgeID target_edge_amount) {
  double probabilty = static_cast<double>(target_edge_amount) / g.m();

  StaticArray<EdgeWeight> sample = StaticArray<EdgeWeight>(g.m());
  for (int v = 0; v < g.n(); ++v) {
    for (EdgeID e : g.incident_edges(v)) {
      KASSERT(v != g.edge_target(e), "no loops allowed", assert::always);
      if (v < g.edge_target(e)) {
        sample[e] = Random::instance().random_bool(probabilty) ? g.edge_weight(e) : 0;
      }
    }
  }
  return sample;
}
} // namespace kaminpar::shm::sparsification