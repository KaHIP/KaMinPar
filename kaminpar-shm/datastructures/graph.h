/*******************************************************************************
 * Wrapper class that delegates all function calls to a concrete graph object.
 *
 * Most function calls are resolved via dynamic binding. Thus, they should not
 * be used when performance is critical. Instead, use an downcast and templatize
 * tight loops.
 *
 * @file:   graph.h
 * @author: Daniel Seemaier
 * @date:   17.11.2023
 ******************************************************************************/
#pragma once

#include <utility>

#include "kaminpar-shm/datastructures/abstract_graph.h"
#include "kaminpar-shm/datastructures/compressed_graph.h"
#include "kaminpar-shm/datastructures/csr_graph.h"
#include "kaminpar-shm/kaminpar.h"

namespace kaminpar::shm {

template <typename Lambda1, typename Lambda2>
decltype(auto) reified(Graph &graph, Lambda1 &&l1, Lambda2 &&l2) {
  AbstractGraph *abstract_graph = graph.underlying_graph();

  if (auto *csr_graph = dynamic_cast<CSRGraph *>(abstract_graph); csr_graph != nullptr) {
    return l1(*csr_graph);
  }

  if (auto *compressed_graph = dynamic_cast<CompressedGraph *>(abstract_graph);
      compressed_graph != nullptr) {
    return l2(*compressed_graph);
  }

  __builtin_unreachable();
}

template <typename Lambda1, typename Lambda2>
decltype(auto) reified(const Graph &graph, Lambda1 &&l1, Lambda2 &&l2) {
  const AbstractGraph *abstract_graph = graph.underlying_graph();

  if (const auto *csr_graph = dynamic_cast<const CSRGraph *>(abstract_graph);
      csr_graph != nullptr) {
    return l1(*csr_graph);
  }

  if (const auto *compressed_graph = dynamic_cast<const CompressedGraph *>(abstract_graph);
      compressed_graph != nullptr) {
    return l2(*compressed_graph);
  }

  __builtin_unreachable();
}

template <typename Lambda> decltype(auto) reified(Graph &graph, Lambda &&l) {
  return reified(graph, std::forward<Lambda>(l), std::forward<Lambda>(l));
}

template <typename Lambda> decltype(auto) reified(const Graph &graph, Lambda &&l) {
  return reified(graph, std::forward<Lambda>(l), std::forward<Lambda>(l));
}

template <typename ConcretizedGraph> [[nodiscard]] ConcretizedGraph &concretize(Graph &graph) {
  KASSERT(
      dynamic_cast<ConcretizedGraph *>(graph.underlying_graph()) != nullptr,
      "underlying graph is not a " << typeid(ConcretizedGraph).name()
  );

  return *static_cast<ConcretizedGraph *>(graph.underlying_graph());
}

template <typename ConcretizedGraph>
[[nodiscard]] const ConcretizedGraph &concretize(const Graph &graph) {
  KASSERT(
      dynamic_cast<const ConcretizedGraph *>(graph.underlying_graph()) != nullptr,
      "underlying graph is not a " << typeid(ConcretizedGraph).name()
  );

  return *static_cast<const ConcretizedGraph *>(graph.underlying_graph());
}

} // namespace kaminpar::shm
