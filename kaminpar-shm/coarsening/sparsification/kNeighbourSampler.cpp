//
// Created by badger on 5/23/24.
//

#include "kNeighbourSampler.h"

#include <networkit/auxiliary/Random.hpp>

#include "kaminpar-common/random.h"

namespace kaminpar::shm::sparsification {

StaticArray<EdgeWeight> kNeighbourSampler::sample(const CSRGraph &g, EdgeID target_edge_amount) {
  int k = target_edge_amount / g.n();

  StaticArray<EdgeWeight> sample = StaticArray<EdgeWeight>(g.m(), 0);
  StaticArray<double> choices = StaticArray<double>(k);
  StaticArray<EdgeWeight> weights_prefix_sum = StaticArray<EdgeWeight>(g.max_degree());

  for (NodeID u : g.nodes()) {
    if (g.degree(u) <= k) {
      for (EdgeID e : g.incident_edges(u)) {
        sample[e] = g.edge_weight(e);
      }
    } else {
      EdgeID first_incident_edge = g.raw_nodes()[u];

      for (int i = 0; i < k; ++i) {
        choices[i] = Aux::Random::real(1.0);
      }
      std::sort(choices.begin(), choices.end());

      weights_prefix_sum[0] = g.edge_weight(first_incident_edge);
      for (int offset = 1; offset < g.degree(u); ++offset) {
        weights_prefix_sum[offset] =
            weights_prefix_sum[offset - 1] + g.edge_weight(first_incident_edge + offset);
      }

      EdgeWeight total_weight = weights_prefix_sum[g.degree(u) - 1];
      EdgeID incident_edge_offset = 0;

      for (int i = 0; i < k; ++i) {
        while (weights_prefix_sum[incident_edge_offset] < choices[i] * total_weight) {
          incident_edge_offset++;
        }
        sample[first_incident_edge + incident_edge_offset] += total_weight / k;
      }
    }
  }

  // TODO: Make more efficent than O(m * n)
  for (NodeID u : g.nodes()) {
    for (EdgeID e : g.incident_edges(u)) {
      NodeID v = g.edge_target(e);
      if (u < v) {
        for (EdgeID f : g.incident_edges(v)) {
          if (g.edge_target(f) == u) {
            sample[e] = (sample[e] + sample[f]) / 2;
          }
        }
      }
    }
  }
}

} // namespace kaminpar::shm::sparsification