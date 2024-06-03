//
// Based on k-Neighbour Sampling form V. Sadhanala, Y.-X. Wang, and R. Tibshirani, “Graph
// Sparsiﬁcation Approaches for Laplacian Smoothing”, 2016
//

#pragma once
#include "Sampler.h"

namespace kaminpar::shm::sparsification {
class kNeighbourSampler : public Sampler {
public:
  kNeighbourSampler(bool sample_spanning_tree = false)
      : _sample_spanning_tree(sample_spanning_tree){};
  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override;

private:
  EdgeID compute_k(const CSRGraph &g, EdgeID target_edge_amount);
  void sample_directed(const CSRGraph &g, EdgeID k, StaticArray<EdgeWeight> &sample);
  void make_sample_symmetric(const CSRGraph &g, StaticArray<EdgeWeight> &sample);
  void sample_spanning_tree(const CSRGraph &g, StaticArray<EdgeWeight> &sample);
  bool _sample_spanning_tree;
};
}

