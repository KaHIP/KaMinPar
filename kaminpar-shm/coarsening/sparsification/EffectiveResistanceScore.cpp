//
// Created by badger on 5/19/24.
//

#include "EffectiveResistanceScore.h"

#include "sparsification_utils.h"
JULIA_DEFINE_FAST_TLS // only define this once, in an executable (not in a shared library) if you
                      // want fast code.

    namespace kaminpar::shm::sparsification {
  void EffectiveResistanceScore::print_jl_exception() {
    jl_value_t *exception = jl_exception_occurred();
    jl_value_t *sprint_fun = jl_get_function(jl_main_module, "sprint");
    jl_value_t *showerror_fun = jl_get_function(jl_main_module, "showerror");

    JL_GC_PUSH3(&exception, &sprint_fun, &showerror_fun);

    if (exception) {
      const char *returned_exception =
          jl_string_ptr(jl_call2(sprint_fun, showerror_fun, exception));
      printf("ERROR: %s\n", returned_exception);
    }

    jl_exception_clear();
    JL_GC_POP();
  }
  EffectiveResistanceScore::IJVMatrix EffectiveResistanceScore::encode_as_ijv(const CSRGraph &g) {
    // Encode ajacency matrix in csc fromat: A[I[n],J[n]] = V[n] and all other entries are zero
    IJVMatrix a = alloc_ijv(g.m());
    utils::for_edges_with_endpoints(g, [&](EdgeID edge, NodeID source, NodeID target) {
      // Julia is 1-indexed
      a.i[edge] = source + 1;
      a.j[edge] = target + 1;
      a.v[edge] = g.edge_weight(edge);
    });
    a.m = g.m();
    return a;
  }

  StaticArray<double> EffectiveResistanceScore::extract_scores(
      const CSRGraph &g, IJVMatrix &sparsifyer
  ) {
    auto endpoint_with_scores = StaticArray<std::tuple<NodeID, NodeID, double>>(sparsifyer.m);
    for (size_t k = 0; k < sparsifyer.m; k++) {
      endpoint_with_scores[k] = std::make_tuple(
          // Back to 0-based indexing
          sparsifyer.i[k] - 1,
          sparsifyer.j[k] - 1,
          sparsifyer.v[k]
      );
    }
    std::sort(
        endpoint_with_scores.begin(),
        endpoint_with_scores.end(),
        [&](const auto &a, const auto &b) {
          return std::get<0>(a) < std::get<0>(b) ||
                 (std::get<0>(a) == std::get<0>(b) && std::get<1>(a) < std::get<1>(b));
        }
    );

    auto scores = StaticArray<double>(g.m(), 0);

    auto sorted_by_target_permutation = utils::sort_by_traget(g);
    for (NodeID u : g.nodes()) {
      for (EdgeID i : g.incident_edges(u)) {
        auto [x, y, w] = endpoint_with_scores[i];

        EdgeID e = sorted_by_target_permutation[i];
        KASSERT(u == x && g.edge_target(e) == y, "julia edges don't match c++ edges");

        scores[e] = w;
      }
    }
    return scores;
  }
  EffectiveResistanceScore::IJVMatrix EffectiveResistanceScore::sparsify_in_julia(IJVMatrix & a) {
    jl_array_t *jl_I = nullptr;
    jl_array_t *jl_J = nullptr;
    jl_array_t *jl_V = nullptr;
    jl_value_t *jl_a = nullptr;

    JL_GC_PUSH4(&jl_I, &jl_I, &jl_V, &jl_a);

    jl_value_t *jl_int_array_type = jl_apply_array_type((jl_value_t *)jl_int64_type, 1);
    jl_I = jl_ptr_to_array_1d(jl_int_array_type, a.i, a.m, 0);
    jl_J = jl_ptr_to_array_1d(jl_int_array_type, a.j, a.m, 0);
    jl_V = jl_ptr_to_array_1d(jl_apply_array_type((jl_value_t *)jl_float64_type, 1), a.v, a.m, 0);

    auto *adapter = (jl_module_t *)jl_eval_string("LapaciansAdapter");

    jl_a = jl_new_struct((jl_datatype_t *)jl_get_function(adapter, "C_IJV"), jl_I, jl_J, jl_V);

    jl_value_t *jl_johnson_lindenstrauss_factor = jl_box_float32(_johnson_lindenstrauss_factor);

    jl_function_t *jl_effective_resistances_function =
        jl_get_function(adapter, "effective_resistances");
    jl_value_t *jl_effective_resistances =
        jl_call2(jl_effective_resistances_function, jl_a, jl_johnson_lindenstrauss_factor);
    print_jl_exception();

    KASSERT(jl_effective_resistances != nullptr, "sparsify_adapter failed!", assert::always);

    IJVMatrix sparsifyer(
        (int64_t *)
            jl_array_data(jl_call1(jl_get_function(adapter, "get_i"), jl_effective_resistances)),
        (int64_t *)
            jl_array_data(jl_call1(jl_get_function(adapter, "get_j"), jl_effective_resistances)),
        (double *)
            jl_array_data(jl_call1(jl_get_function(adapter, "get_v"), jl_effective_resistances)),
        jl_unbox_int64(jl_call1(jl_get_function(adapter, "get_m"), jl_effective_resistances))
    );

    JL_GC_POP();

    print_jl_exception();
    return sparsifyer;
  }

  StaticArray<double> EffectiveResistanceScore::scores(const CSRGraph &g) {
    IJVMatrix a = encode_as_ijv(g);
    IJVMatrix sparsifyer = sparsify_in_julia(a);
    free_ijv(a);
    auto sample = extract_scores(g, sparsifyer);
    // The sample is freed by the julia garbage collector (hopefully)
    return sample;
  }

  EffectiveResistanceScore::IJVMatrix EffectiveResistanceScore::alloc_ijv(EdgeID m) {
    return IJVMatrix(
        (int64_t *)malloc(sizeof(int64_t) * m),
        (int64_t *)malloc(sizeof(int64_t) * m),
        (double *)malloc(sizeof(double) * m),
        m
    );
  }
  void EffectiveResistanceScore::free_ijv(IJVMatrix & a) {
    free(a.i);
    free(a.j);
    free(a.v);
  }

} // namespace kaminpar::shm::sparsification