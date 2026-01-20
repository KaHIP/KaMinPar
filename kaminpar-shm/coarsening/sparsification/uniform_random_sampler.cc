#include "kaminpar-shm/coarsening/sparsification/uniform_random_sampler.h"

#include "kaminpar-common/random.h"

namespace kaminpar::shm::sparsification {

StaticArray<EdgeWeight>
UniformRandomSampler::sample(const CSRGraph &g, const EdgeID target_edge_amount) {
  const double probabilty = static_cast<double>(target_edge_amount) / g.m();

  StaticArray<EdgeWeight> sample(g.m());
  for (NodeID v = 0; v < g.n(); ++v) {
    for (EdgeID e : g.incident_edges(v)) {
      KASSERT(v != g.edge_target(e), "no loops allowed", assert::always);
      if (v < g.edge_target(e)) {
        sample[e] = Random::instance().random_bool(probabilty) ? g.edge_weight(e) : 0;
      }
    }
  }
  return sample;
}

StaticArray<EdgeWeight>
WeightedUniformRandomSampler::sample(const CSRGraph &g, const EdgeID desired_num_edges) {
  const double probability = static_cast<double>(desired_num_edges) / g.m();

  StaticArray<EdgeWeight> sample(g.m());
  for (NodeID v = 0; v < g.n(); ++v) {
    for (EdgeID e : g.incident_edges(v)) {
      KASSERT(v != g.edge_target(e), "no loops allowed", assert::always);
      if (v < g.edge_target(e)) {
        sample[e] = Random::instance().random_bool(probability) ? g.edge_weight(e) : 0;
      }
    }
  }
  return sample;
}

} // namespace kaminpar::shm::sparsification
