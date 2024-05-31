//
// Created by badger on 5/19/24.
//

#pragma once

#include <julia.h>

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
  IJV alloc_ijv(EdgeID m);
  void free_ijv(IJV &a);
  void print_jl_exception();
  IJV encode_as_ijv(const CSRGraph &g);
  StaticArray<EdgeWeight> extract_sample(const CSRGraph &g, IJV &sparsifyer);
  IJV sparsify_in_julia(IJV &a);

  inline static const char *JL_LAPLACIANS_ADAPTER_CODE = R"(
  module LapaciansAdapter
    using Laplacians
    using SparseArrays

    struct C_IJV
        i::Array{Int64,1}
        j::Array{Int64,1}
        v::Array{Cdouble,1}
    end

    function sparsify_adapter(A::C_IJV, eps::Cfloat)::C_IJV
        sparsifyed = sparsify(sparse(A.i, A.j, A.v),ep = eps)
        (i, j, v) = findnz(sparsifyed)
        return C_IJV(i, j, v)
    end

    function get_i(A::C_IJV)
        return A.i
    end
    function get_j(A::C_IJV)
        return A.j
    end
    function get_v(A::C_IJV)
        return A.v
    end
    function get_m(A::C_IJV)
        return length(A.i)
    end

    function get_first(arr)
      return arr[1]
    end
  end
  )";
};
} // namespace kaminpar::shm::sparsification
