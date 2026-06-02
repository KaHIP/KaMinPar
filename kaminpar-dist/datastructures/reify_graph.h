/*******************************************************************************
 * Implements the explicit downcasts of abstract graphs to concrete graphs.
 *
 * @file:   reify_graph.h
 * @author: Daniel Seemaier
 * @date:   25.12.2024
 ******************************************************************************/
#pragma once

#include "kaminpar-dist/datastructures/abstract_distributed_graph.h"
#include "kaminpar-dist/datastructures/distributed_compressed_graph.h"
#include "kaminpar-dist/datastructures/distributed_csr_graph.h"

#include "kaminpar-common/assert.h"

namespace kaminpar::dist::graph {

template <typename Lambda1, typename Lambda2>
decltype(auto) reified(const AbstractDistributedGraph &graph, Lambda1 &&l1, Lambda2 &&l2) {
  if (auto *csr_graph = dynamic_cast<const DistributedCSRGraph *>(&graph); csr_graph != nullptr) {
    return l1(*csr_graph);
  } else if (auto *compressed_graph = dynamic_cast<const DistributedCompressedGraph *>(&graph);
             compressed_graph != nullptr) {
    return l2(*compressed_graph);
  }

  __builtin_unreachable();
}

template <typename Lambda>
decltype(auto) reified(const AbstractDistributedGraph &graph, Lambda &&l) {
  if (auto *csr_graph = dynamic_cast<const DistributedCSRGraph *>(&graph); csr_graph != nullptr) {
    return l(*csr_graph);
  } else if (auto *compressed_graph = dynamic_cast<const DistributedCompressedGraph *>(&graph);
             compressed_graph != nullptr) {
    return l(*compressed_graph);
  }

  __builtin_unreachable();
}

template <typename ConcreteGraph> [[nodiscard]] bool is(const AbstractDistributedGraph &graph) {
  return dynamic_cast<const ConcreteGraph *>(&graph) != nullptr;
}

template <typename ConcreteGraph>
[[nodiscard]] ConcreteGraph &as_concrete_graph(const AbstractDistributedGraph &graph) {
  KASSERT(
      is<ConcreteGraph>(graph), "underlying graph is not a " << typeid(ConcreteGraph).name()
  );
  return *static_cast<ConcreteGraph *>(&graph);
}

template <typename ConcreteGraph> ConcreteGraph &as_concrete_graph(AbstractDistributedGraph &graph) {
  KASSERT(
      is<ConcreteGraph>(graph), "underlying graph is not a " << typeid(ConcreteGraph).name()
  );
  return dynamic_cast<ConcreteGraph &>(graph);
}

} // namespace kaminpar::dist::graph
