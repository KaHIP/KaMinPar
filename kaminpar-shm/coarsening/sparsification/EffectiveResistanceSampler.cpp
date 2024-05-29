//
// Created by badger on 5/19/24.
//

#include "EffectiveResistanceSampler.h"

#include <networkit/algebraic/Vector.hpp>

namespace kaminpar::shm::sparsification {
EffectiveResistanceSampler::EffectiveResistanceSampler() {
  jl_init();
}

EffectiveResistanceSampler::~EffectiveResistanceSampler() {
  jl_atexit_hook(0);
}

void EffectiveResistanceSampler::print_jl_exception() {
  jl_value_t *exception = jl_exception_occurred();
  jl_value_t *sprint_fun = jl_get_function(jl_main_module, "sprint");
  jl_value_t *showerror_fun = jl_get_function(jl_main_module, "showerror");

  JL_GC_PUSH3(&exception, &sprint_fun, &showerror_fun);

  if (exception) {
    const char *returned_exception = jl_string_ptr(jl_call2(sprint_fun, showerror_fun, exception));
    printf("ERROR: %s\n", returned_exception);
  }

  jl_exception_clear();
  JL_GC_POP();
}
EffectiveResistanceSampler::IJV *EffectiveResistanceSampler::encode_as_ijv(const CSRGraph &g) {
  // Encode ajacency matrix in csc fromat: A[I[n],J[n]] = V[n] and all other entries are zero
  IJV *a = (IJV *)malloc(sizeof(IJV));
  a->i = (int64_t *)malloc(sizeof(int64_t) * g.m());
  a->j = (int64_t *)malloc(sizeof(int64_t) * g.m());
  a->v = (double *)malloc(sizeof(double) * g.m());
  for (NodeID source : g.nodes()) {
    for (EdgeID edge : g.incident_edges(source)) {
      NodeID target = g.edge_target(edge);
      a->i[edge] = source + 1;
      a->j[edge] = target + 1;
      a->v[edge] = g.edge_weight(edge);
    }
  }
  a->m = g.m();
  return a;
}

StaticArray<EdgeWeight>
EffectiveResistanceSampler::extract_sample(const CSRGraph &g, IJV *sparsifyer) {
  auto sampled_edges = StaticArray<std::tuple<NodeID, NodeID, EdgeWeight>>(sparsifyer->m);
  for (size_t k = 0; k < sparsifyer->m; k++) {
    sampled_edges[k] = std::make_tuple(
        sparsifyer->i[k] - 1, sparsifyer->j[k] - 1, static_cast<EdgeWeight>(sparsifyer->v[k])
    );
  }
  std::sort(sampled_edges.begin(), sampled_edges.end(), [&](const auto &a, const auto &b) {
    return std::get<0>(a) < std::get<0>(b) ||
           (std::get<0>(a) == std::get<0>(b) && std::get<1>(a) < std::get<1>(b));
  });
  auto sample = StaticArray<EdgeWeight>(g.m(), 0);

  auto sorted_by_target_permutation = StaticArray<EdgeID>(g.m());
  for (EdgeID e : g.edges())
    sorted_by_target_permutation[e] = e;
  for (NodeID u : g.nodes()) {
    std::sort(
        sorted_by_target_permutation.begin() + g.raw_nodes()[u],
        sorted_by_target_permutation.begin() + g.raw_nodes()[u + 1],
        [&](const EdgeID e1, const EdgeID e2) { return g.edge_target(e1) <= g.edge_target(e2); }
    );
  }
  size_t k = 0;
  for (NodeID u : g.nodes()) {
    if (k == sparsifyer->m)
      break;
    for (EdgeID i : g.incident_edges(u)) {
      EdgeID e = sorted_by_target_permutation[i];
      NodeID v = g.edge_target(e);
      auto [x, y, w] = sampled_edges[k];
      if (u == x && v == y) {
        sample[e] = w;
        k++;
      }
    }
  }
  KASSERT(k == sparsifyer->m, "Not alle sampled edges were added to sample!", assert::always);
  return sample;
}
EffectiveResistanceSampler::IJV *EffectiveResistanceSampler::sparsify_in_julia(IJV *a) {
  jl_eval_string(R"(
  module Adapter
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
  )");

  // TODO: Enable GC carefully
  JL_GC_DISABLED(1);

  jl_value_t *jl_int_array_type = jl_apply_array_type((jl_value_t *)jl_int64_type, 1);
  jl_array_t *jl_I = jl_ptr_to_array_1d(jl_int_array_type, a->i, a->m, 0);
  jl_array_t *jl_J = jl_ptr_to_array_1d(jl_int_array_type, a->j, a->m, 0);
  jl_array_t *jl_V =
      jl_ptr_to_array_1d(jl_apply_array_type((jl_value_t *)jl_float64_type, 1), a->v, a->m, 0);

  jl_module_t *adapter = (jl_module_t *)jl_eval_string("Adapter");

  jl_value_t *jl_a =
      jl_new_struct((jl_datatype_t *)jl_get_function(adapter, "C_IJV"), jl_I, jl_J, jl_V);

  // TODO: Pick eps in a sensible way
  jl_value_t *jl_eps = jl_box_float32(0.1);

  jl_function_t *sparsify_adapter = jl_get_function(adapter, "sparsify_adapter");
  jl_value_t *jl_sparsifyer = jl_call2(sparsify_adapter, jl_a, jl_eps);
  print_jl_exception();

  KASSERT(jl_sparsifyer != nullptr, "sparsify_adapter failed!", assert::always);

  IJV *sparsifyer = (IJV *)malloc(sizeof(IJV));
  sparsifyer->i =
      (int64_t *)jl_array_data(jl_call1(jl_get_function(adapter, "get_i"), jl_sparsifyer));
  sparsifyer->j =
      (int64_t *)jl_array_data(jl_call1(jl_get_function(adapter, "get_j"), jl_sparsifyer));
  sparsifyer->v =
      (double *)jl_array_data(jl_call1(jl_get_function(adapter, "get_v"), jl_sparsifyer));
  sparsifyer->m = jl_unbox_uint32(jl_call1(jl_get_function(adapter, "get_m"), jl_sparsifyer));

  print_jl_exception();
  return sparsifyer;
}

StaticArray<EdgeWeight>
EffectiveResistanceSampler::sample(const CSRGraph &g, EdgeID target_edge_amount) {
  // TODO: Free memory and maybe reduce use of C-style pointer
  IJV *a = encode_as_ijv(g);

  IJV * sparsifyer = sparsify_in_julia(a);

  return extract_sample(g, sparsifyer);
}
} // namespace kaminpar::shm::sparsification