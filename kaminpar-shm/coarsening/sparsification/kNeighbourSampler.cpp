//
// Created by badger on 5/23/24.
//

#include "kNeighbourSampler.h"

#include <networkit/auxiliary/Random.hpp>

#include "kaminpar-common/random.h"

namespace kaminpar::shm::sparsification {

StaticArray<EdgeWeight> kNeighbourSampler::sample(const CSRGraph &g, EdgeID target_edge_amount) {
  int k = compute_k(g, target_edge_amount);
  StaticArray<EdgeWeight> sample = sample_directed(g, k);
  make_sample_symmetric(g, sample);
  return sample;
}

/*
 * compute max k s.t. the number of sampled edges is at most target_edge_amount
 */
EdgeID kNeighbourSampler::compute_k(const CSRGraph &g, EdgeID target_edge_amount) {
  auto nodes_of_degree = StaticArray<EdgeID>(g.max_degree());
  for (NodeID u : g.nodes()) {
    nodes_of_degree[g.degree(u)]++;
  }

  EdgeID k = 0, incidence_with_degree_lt_k = 0;
  NodeID nodes_with_degree_gt_k = g.n();
  while (target_edge_amount <= incidence_with_degree_lt_k + k * nodes_with_degree_gt_k) {
    incidence_with_degree_lt_k += k * nodes_of_degree[k];
    nodes_with_degree_gt_k -= nodes_of_degree[k];
    k++;
  }
  return k - 1;
}

StaticArray<EdgeWeight> kNeighbourSampler::sample_directed(const CSRGraph &g, EdgeID k) {
  StaticArray<EdgeWeight> sample = StaticArray<EdgeWeight>(g.m(), 0);
  StaticArray<double> choices = StaticArray<double>(k);
  StaticArray<EdgeWeight> weights_prefix_sum = StaticArray<EdgeWeight>(g.max_degree());

  // First choose locally at evey node how much of each incident edge to sample
  for (NodeID u : g.nodes()) {
    if (g.degree(u) <= k) { // sample all edges
      for (EdgeID e : g.incident_edges(u)) {
        sample[e] = g.edge_weight(e);
      }
    } else { // sample k incicdent edges randomly, with probailties proportional to edge weights
      EdgeID first_incident_edge = g.raw_nodes()[u];
      weights_prefix_sum[0] = g.edge_weight(first_incident_edge);
      for (int offset = 1; offset < g.degree(u); ++offset) {
        weights_prefix_sum[offset] =
            weights_prefix_sum[offset - 1] + g.edge_weight(first_incident_edge + offset);
      }
      EdgeWeight total_weight = weights_prefix_sum[g.degree(u) - 1];

      for (int i = 0; i < k; ++i) {
        choices[i] = Aux::Random::real(total_weight);
      }
      std::sort(choices.begin(), choices.end());

      EdgeID incident_edge_offset = 0;
      for (int i = 0; i < k; ++i) {
        while (weights_prefix_sum[incident_edge_offset] < choices[i]) {
          incident_edge_offset++;
        }
        sample[first_incident_edge + incident_edge_offset] += total_weight / k;
      }
    }
  }
}

void kNeighbourSampler::make_sample_symmetric(const CSRGraph &g, StaticArray<EdgeWeight> &sample) {
  // Then combine the sample of each edge at both endpoints
  StaticArray<EdgeID> sorted_by_target_permutation = StaticArray<EdgeID>(g.m());
  for (auto e : g.edges())
    sorted_by_target_permutation[e] = e;
  for (NodeID u : g.nodes()) {
    std::sort(
        sorted_by_target_permutation.begin() + g.raw_nodes()[u],
        sorted_by_target_permutation.begin() + g.raw_nodes()[u + 1],
        [&](const auto e1, const auto e2) { return g.edge_target(e1) <= g.edge_target(e2); }
    );
  }

  auto edges_done = StaticArray<EdgeID>(g.n(), 0);
  for (NodeID u : g.nodes()) {
    for (EdgeID e : g.incident_edges(u)) {
      NodeID v = g.edge_target(e);
      if (u < v) {
        EdgeID counter_edge = sorted_by_target_permutation[g.raw_nodes()[v] + edges_done[v]];
        KASSERT(g.edge_target(counter_edge) == u, "Graph was not sorted", assert::always);
        EdgeWeight combined_sample = (sample[e] + sample[counter_edge]) / 2;
        sample[e] = combined_sample;
        sample[counter_edge] = combined_sample;
        edges_done[v]++;
      }
    }
  }
}

} // namespace kaminpar::shm::sparsification