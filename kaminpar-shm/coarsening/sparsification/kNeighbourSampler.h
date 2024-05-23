//
// Based on k-Neighbour Sampling form V. Sadhanala, Y.-X. Wang, and R. Tibshirani, “Graph
// Sparsiﬁcation Approaches for Laplacian Smoothing”, 2016
//

#pragma once
#include "Sampler.h"

namespace kaminpar::shm::sparsification {
class kNeighbourSampler : public Sampler{
  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override;
};
}

