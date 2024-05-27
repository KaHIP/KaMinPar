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

struct IJV {
  EdgeID *i;
  EdgeID *j;
  double *v;
  NodeID n;
};

StaticArray<EdgeWeight>
EffectiveResistanceSampler::sample(const CSRGraph &g, EdgeID target_edge_amount) {
  // Encode ajacency matrix in csc fromat: A[I[n],J[n]] = V[n] and all other entries are zero
  IJV *a = (IJV *)malloc(sizeof(IJV));
  a->i = (EdgeID *)malloc(sizeof(EdgeID) * g.m());
  a->j = (EdgeID *)malloc(sizeof(EdgeID) * g.m()), a->v = (double *)malloc(sizeof(double) * g.m());
  for (NodeID source : g.nodes()) {
    for (EdgeID edge : g.incident_edges(source)) {
      NodeID target = g.edge_target(edge);
      a->i[edge] = source;
      a->j[edge] = target;
      a->v[edge] = g.edge_weight(edge);
    }
  }

  jl_eval_string(R"(
  module Adapter
    using Laplacians
    using SparseArrays

    struct C_IJV
        i::Array{Cint,1}
        j::Array{Cint,1}
        v::Array{Cdouble,1}
    end

    function sparsify_adapter(A::C_IJV, eps::Cfloat)::C_IJV
        sparsifyed = sparsify(sparse(A.i, A.j, A.v),ep = eps)
        (i, j, v) = findnz(sparsifyed)
        return C_IJV(i, j, v)
    end
  end
  )");

  jl_value_t *jl_int_array_type = jl_apply_array_type((jl_value_t *)jl_int32_type, 1);
  jl_array_t *jl_I = jl_ptr_to_array_1d(jl_int_array_type, a->i, g.m(), 1);
  jl_array_t *jl_J = jl_ptr_to_array_1d(jl_int_array_type, a->j, g.m(), 1);
  jl_array_t *jl_V =
      jl_ptr_to_array_1d(jl_apply_array_type((jl_value_t *)jl_float64_type, 1), a->v, g.m(), 1);

  jl_module_t *adapter = (jl_module_t *)jl_eval_string("Adapter");
  /*
  jl_value_t *jl_a = jl_new_struct(
      (jl_datatype_t *)jl_get_function(adapter, "C_IJV"), jl_I, jl_J, jl_V
  );
  */
  jl_function_t *jl_ijv = jl_get_function(adapter, "C_IJV");
  jl_value_t *jl_a = jl_call3(jl_ijv, (jl_value_t *)jl_I, (jl_value_t *)jl_J, (jl_value_t *)jl_V);
  jl_function_t *sparsify_adapter = jl_get_function(adapter, "sparsify_adapter");
  KASSERT(
      sparsify_adapter != nullptr,
      "Could not find function sparsify_adapter in Julia.",
      assert::always
  );

  jl_value_t *jl_eps = jl_box_float32(0.7);

  KASSERT(jl_is_structtype(jl_a), "jl_a is not a struct", assert::always);

  jl_value_t *jl_sparsifier = jl_call2(sparsify_adapter, jl_a, jl_eps);

  if (jl_exception_occurred())
    printf(
        "%s: %s \n", jl_typeof_str(jl_exception_occurred()), jl_typeof_str(jl_current_exception())
    );
  return StaticArray<EdgeWeight>(g.m());
}
} // namespace kaminpar::shm::sparsification