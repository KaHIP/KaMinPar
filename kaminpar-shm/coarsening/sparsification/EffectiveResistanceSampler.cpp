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

StaticArray<EdgeWeight>
EffectiveResistanceSampler::sample(const CSRGraph &g, EdgeID target_edge_amount) {
  // Encode ajacency matrix in csc fromat: A[I[n],J[n]] = V[n] and all other entries are zero
  EdgeWeight *I = (EdgeWeight *)malloc(sizeof(EdgeWeight) * g.m()),
             *J = (EdgeWeight *)malloc(sizeof(EdgeWeight) * g.m()),
             *V = (EdgeWeight *)malloc(sizeof(EdgeWeight) * g.m());
  for (NodeID source : g.nodes()) {
    for (EdgeID edge : g.incident_edges(source)) {
      NodeID target = g.edge_target(edge);
      I[edge] = source;
      J[edge] = target;
      V[edge] = g.edge_weight(edge);
    }
  }

  jl_value_t *array_type = jl_apply_array_type((jl_value_t *)jl_int32_type, 1);
  jl_array_t *jl_I = jl_ptr_to_array_1d(array_type, I, g.m(), 1);
  jl_array_t *jl_J = jl_ptr_to_array_1d(array_type, J, g.m(), 1);
  jl_array_t *jl_V = jl_ptr_to_array_1d(array_type, V, g.m(), 1);

  jl_eval_string("cd(\"/home/badger/Uni/S6/BA/KaMinPar/Laplacians.jl/src/\");include(\"Laplacians.jl\")");


  jl_module_t *laplacian = (jl_module_t *)jl_eval_string("Main.Laplacians");
  if (jl_exception_occurred())
    printf("%s: %s \n", jl_typeof_str(jl_exception_occurred()), jl_typeof_str(jl_current_exception()));
  KASSERT(laplacian != nullptr, "Could not find module Laplacian in Julia.", assert::always);


  jl_function_t *sparse = jl_get_function(laplacian, "sparse");
  jl_function_t *sparsify = jl_get_function(laplacian, "sparsify");
  jl_function_t *nnz = jl_get_function(laplacian, "sparsify");
  jl_function_t *findnz = jl_get_function(laplacian, "sparsify");
  KASSERT(sparse != nullptr, "Could not find function sparse in Julia.", assert::always);
  KASSERT(sparsify != nullptr, "Could not find function sparsfiy in Julia.", assert::always);

  jl_value_t *adjacency_matrix =
      jl_call3(sparse, (jl_value_t *)jl_I, (jl_value_t *)jl_J, (jl_value_t *)jl_V);

  jl_value_t *sparsifyer = jl_call1(sparsify, adjacency_matrix);

  if (jl_exception_occurred())
    printf("%s: %s \n", jl_typeof_str(jl_exception_occurred()), jl_typeof_str(jl_current_exception()));
  return StaticArray<EdgeWeight>(g.m());
}
} // namespace kaminpar::shm::sparsification