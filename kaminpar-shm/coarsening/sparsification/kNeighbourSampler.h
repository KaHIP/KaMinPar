//
// Based on k-Neighbour Sampling form V. Sadhanala, Y.-X. Wang, and R. Tibshirani, “Graph
// Sparsiﬁcation Approaches for Laplacian Smoothing”, 2016
//

#pragma once
#include "Sampler.h"

namespace kaminpar::shm::sparsification {
class kNeighbourSampler : public Sampler {
public:
  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override;
private:
  EdgeID compute_k(const CSRGraph &g, EdgeID target_edge_amount);
  StaticArray<EdgeWeight> sample_directed(const CSRGraph &g, EdgeID k);
  void make_sample_symmetric(const CSRGraph &g, StaticArray<EdgeWeight> &sample);
};
}

