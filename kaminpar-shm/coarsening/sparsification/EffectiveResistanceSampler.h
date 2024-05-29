//
// Created by badger on 5/19/24.
//

#pragma once

#include <julia.h>
// JULIA_DEFINE_FAST_TLS // only define this once, in an executable (not in a shared library) if you
// want fast code.

#include "Sampler.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm::sparsification {
class EffectiveResistanceSampler : public Sampler {
public:
  EffectiveResistanceSampler();
  ~EffectiveResistanceSampler();
  StaticArray<EdgeWeight> sample(const CSRGraph &g, EdgeID target_edge_amount) override;

private:
  struct IJV {
    int64_t *i;
    int64_t *j;
    double *v;
    EdgeID m;
  };
  void print_jl_exception();
  IJV *encode_as_ijv(const CSRGraph &g);
  StaticArray<EdgeWeight> extract_sample(const CSRGraph &g, IJV *sparsifyer);
  IJV *sparsify_in_julia(IJV *a);
};
} // namespace kaminpar::shm::sparsification
