//
// Created by badger on 5/23/24.
//

#include "kNeighbourSampler.h"

#include <networkit/auxiliary/Random.hpp>

#include "IndexDistributionWithReplacement.h"
#include "IndexDistributionWithoutReplacement.h"
#include "UnionFind.h"
#include "sparsification_utils.h"

#include "kaminpar-common/parallel/algorithm.h"
#include "kaminpar-common/random.h"

namespace kaminpar::shm::sparsification {

StaticArray<EdgeWeight> kNeighbourSampler::sample(const CSRGraph &g, EdgeID target_edge_amount) {
  int k = compute_k(g, target_edge_amount);
  auto sample = StaticArray<EdgeWeight>(g.m(), 0);
  if (_sample_spanning_tree)
    sample_spanning_tree(g, sample);
  sample_directed(g, k, sample);
  make_sample_symmetric(g, sample);
  return sample;
}

/*
 * compute max k s.t. the number of sampled edges is at most target_edge_amount
 */
EdgeID kNeighbourSampler::compute_k(const CSRGraph &g, EdgeID target_edge_amount) {
  StaticArray<EdgeID> edges_incident_to_smaller_degree(g.n() - 1, 0);
  utils::for_edges_with_endpoints(g, [&](EdgeID e, NodeID u, NodeID v) {
    edges_incident_to_smaller_degree[std::max(g.degree(u), g.degree(v))];
  });
  parallel::prefix_sum(
      edges_incident_to_smaller_degree.begin(),
      edges_incident_to_smaller_degree.end(),
      edges_incident_to_smaller_degree.begin()
  );
  StaticArray<EdgeWeight> incident_weights(g.n(), 0);
  utils::for_edges_with_endpoints(g, [&](EdgeID e, NodeID u, NodeID v) {
    incident_weights[u] += g.edge_weight(e);
    incident_weights[v] += g.edge_weight(e);
  });
  auto expected_m = [&](NodeID k) {
    // Exp(m) = |{u v in E : deg(u)<=k or deg(v) <= k}|
    //        + sum_(u in V, deg(u)>k) sum_(v in N(u), deg(v)>k) 1 - (1-w_(uv)/(W_u W_v))
    // With w_(uv) being the weight of the edge beween u and v and W_u being the incident weights of
    // vertex u
    double sum = 0;
    for (NodeID u : g.nodes()) {
      if (g.degree(u) <= k)
        continue;
      for (EdgeID e : g.incident_edges(u)) {
        NodeID v = g.edge_target(e);
        if (g.degree(v) <= k)
          continue;
        sum += 1 - (1 - g.edge_weight(e) / pow(incident_weights[u] * incident_weights[v], k));
      }
    }
    return edges_incident_to_smaller_degree[k] + sum;
  };

  // binary search
  NodeID k = 1;
  NodeID low = 1;
  NodeID heigh = target_edge_amount / g.n() + 1;
  while (low + 1 != heigh) {
    k = low + (heigh - low) / 2;
    double expected = expected_m(k);
    if (target_edge_amount < expected)
      heigh = k;
    else
      low = k;
  }
  return k;
}

void kNeighbourSampler::sample_directed(
    const CSRGraph &g, EdgeID k, StaticArray<EdgeWeight> &sample
) {
  StaticArray<double> choices = StaticArray<double>(k);
  StaticArray<EdgeWeight> weights_prefix_sum = StaticArray<EdgeWeight>(g.max_degree());

  // First choose locally at evey node how much of each incident edge to sample
  for (NodeID u : g.nodes()) {
    if (g.degree(u) <= k) { // sample all edges
      for (EdgeID e : g.incident_edges(u)) {
        sample[e] = g.edge_weight(e);
      }
    } else { // sample k incicdent edges randomly, with probailties proportional to edge weights
      IndexDistributionWithReplacement distribution(
          g.raw_edge_weights().begin() + g.raw_nodes()[u],
          g.raw_edge_weights().begin() + g.raw_nodes()[u + 1]
      );

      EdgeWeight total_weight = std::reduce(
          g.raw_edge_weights().begin() + g.raw_nodes()[u],
          g.raw_edge_weights().begin() + g.raw_nodes()[u + 1],
          0, std::plus<>()
      );

      for (int i = 0; i < k; ++i) {
        sample[distribution()] += total_weight / k;
      }
    }
  }
}

void kNeighbourSampler::make_sample_symmetric(const CSRGraph &g, StaticArray<EdgeWeight> &sample) {
  // Then combine the sample of each edge at both endpoints
  StaticArray<EdgeID> sorted_by_target_permutation = utils::sort_by_traget(g);

  auto edges_done = StaticArray<EdgeID>(g.n(), 0);
  for (NodeID u : g.nodes()) {
    for (EdgeID e : g.incident_edges(u)) {
      NodeID v = g.edge_target(e);
      if (u < v) {
        EdgeID counter_edge = sorted_by_target_permutation[g.raw_nodes()[v] + edges_done[v]];
        KASSERT(g.edge_target(counter_edge) == u, "incorrect counter_edge", assert::always);
        EdgeWeight combined_sample = (sample[e] + sample[counter_edge]) / 2;
        sample[e] = combined_sample;
        sample[counter_edge] = combined_sample;
        edges_done[v]++;
      }
    }
  }
}

void kNeighbourSampler::sample_spanning_tree(const CSRGraph &g, StaticArray<EdgeWeight> &sample) {
  // Kruskal's algorithm
  auto edges = StaticArray<std::tuple<NodeID, EdgeID>>(g.m());
  for (NodeID u : g.nodes()) {
    for (EdgeID e : g.incident_edges(u)) {
      edges[e] = std::make_tuple(u, e);
    }
  }
  std::sort(edges.begin(), edges.end(), [&](const auto &e1, const auto &e2) {
    return 1.0 / g.edge_weight(std::get<1>(e1)) < 1.0 / g.edge_weight(std::get<1>(e2));
  });

  auto uf = UnionFind<NodeID>(g.n());
  for (auto [u, e] : edges) {
    NodeID v = g.edge_target(e);
    if (uf.find(u) != uf.find(v)) {
      uf.unionNodes(u, v);
      sample[e] = g.edge_weight(e);
    }
  }
}

} // namespace kaminpar::shm::sparsification