//
// Created by badger on 5/19/24.
//

#pragma once

#include <julia.h>

#include "Sampler.h"
#include "ScoreBacedSampler.h"

#include "kaminpar-shm/datastructures/graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm::sparsification {
class EffectiveResistanceScore : public ScoreFunction<double> {
public:
  static void init_julia() {
    jl_init();
    jl_eval_string(JL_LAPLACIANS_ADAPTER_CODE);
    print_jl_exception();
  }
  static void finalize_julia() {
    jl_atexit_hook(0);
  }
  static void print_jl_exception();

  EffectiveResistanceScore(float johnson_lindenstrauss_factor)
      : _johnson_lindenstrauss_factor(johnson_lindenstrauss_factor) {}

  StaticArray<double> scores(const CSRGraph &g) override;

private:
  struct IJVMatrix {
    int64_t *i;
    int64_t *j;
    double *v;
    EdgeID m;
  };
  float _johnson_lindenstrauss_factor;
  IJVMatrix alloc_ijv(EdgeID m);
  void free_ijv(IJVMatrix &a);

  IJVMatrix encode_as_ijv(const CSRGraph &g);
  StaticArray<double> extract_scores(const CSRGraph &g, IJVMatrix &sparsifyer);
  IJVMatrix sparsify_in_julia(IJVMatrix &a);

  inline static const char *JL_LAPLACIANS_ADAPTER_CODE = R"(
  module LapaciansAdapter
    using Laplacians
    using SparseArrays
    using LinearAlgebra

    struct C_IJV
        i::Array{Int64,1}
        j::Array{Int64,1}
        v::Array{Float64,1}
    end

    function get_i(a::C_IJV)::Array{Int64}
      return a.i
    end
    function get_j(a::C_IJV)::Array{Int64}
      return a.j
    end
    function get_v(a::C_IJV)::Array{Float64}
      return a.v
    end
    function get_m(a::C_IJV)::Int64
      return length(a.v)
    end

    # based on sparsify method in module Lapacians
    function effective_resistances(g::C_IJV, JLfac::Float32)::C_IJV
      a = sparse(g.i, g.j, g.v)
      f = approxchol_lap(a,tol=1e-2);

      n = size(a,1)
      k = round(Int, JLfac*log(n)) # number of dims for Johnson-Lindenstrauss

      U = wtedEdgeVertexMat(a)
      m = size(U,1)
      R = randn(Float64, m,k)
      UR = U'*R;

      V = zeros(n,k)
      for i in 1:k
        V[:,i] = f(UR[:,i])
      end

      (ai,aj,av) = findnz(triu(a))
      ers = zeros(size(av))
      for h in 1:length(av)
        i = ai[h]
        j = aj[h]
        # divided by k, because 1/sqrt(k)
        ers[h] = min(1,av[h]* ((norm(V[i,:]-V[j,:])^2)/k))
        #        min(1, w_e * (R_e                      ))
      end
      erMatrix = sparse(ai,aj,ers,n,n)
      erMatrix = erMatrix + erMatrix'

      (i,j,v) = findnz(erMatrix)
      return C_IJV(i,j,v)
    end
  end
  )";
};
} // namespace kaminpar::shm::sparsification
